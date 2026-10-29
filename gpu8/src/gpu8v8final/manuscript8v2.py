"""Revised V8.1 manuscript (2026-09-26): expanded experimental set-up, figure-driven results with
full legends, post-hoc descriptive analyses clearly separated from the registered inference.

Every number is read from the saved results (numbers_v8_results.json, audits) or from the source-data
tables that the publication figures were drawn from (manuscript_v8/figures_v2/source_data), so the
text, the tables and the figures cannot disagree.  The 2026-09-24 manuscript is left untouched.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import paths8
from .common8 import dump_json, now_utc
from .manuscript8 import ci, e5_success, f, g, method_row, pc, table1, table3, table4, table_md
from .mstext8 import METHODS

OUT = paths8.MANUSCRIPT
DATE = "20260926"
FIGDIR = OUT / "figures_v2"
SD = FIGDIR / "source_data"
PUB = paths8.RESULTS / "pub"


# ----------------------------------------------------------------------------- loading
def sd(stem: str, panel: str) -> pd.DataFrame:
    return pd.read_csv(SD / f"{stem}_{panel}.csv", encoding="utf-8-sig")


def load_R() -> dict:
    from . import narrative8
    R = json.loads((paths8.RESULTS / "numbers_v8_results.json").read_text(encoding="utf-8"))
    R["strata"] = json.loads((paths8.RESULTS / "analysis" / "strata_zero_signal_v8.json").read_text(encoding="utf-8"))
    R["hitflow"] = json.loads((paths8.RESULTS / "analysis" / "hitflow_s42_v8.json").read_text(encoding="utf-8"))
    acc = json.loads((paths8.AUDIT / "acceptance_ten_checks_v8.json").read_text(encoding="utf-8"))
    R["fallback"] = acc.get("8_fallback", {})
    sels = [json.loads((paths8.DEV / f"fold{j}" / "selection.json").read_text(encoding="utf-8")) for j in range(5)]
    R["dev"] = {"v6_cov": float(np.mean([s["w"]["coverage"]["v6_500"] for s in sels])),
                "strict_cov": float(np.mean([s["w"]["coverage"][f"aware_w{s['w']['w']}"] for s in sels]))}
    R["_txt_inputs"] = narrative8.inputs_text()
    R["_txt_dev"] = narrative8.dev_text()
    R["_txt_discussion"] = narrative8.discussion_text(R)
    R["_txt_limits"] = narrative8.limits_text(R)
    R["_txt_conclusion"] = narrative8.conclusion_text(R)
    R["_txt_e9"] = narrative8.e9_text(R)
    return R


def manifest() -> dict:
    return json.loads((FIGDIR / "figures_manifest.json").read_text(encoding="utf-8"))


def fig(stem: str, n: str, title: str, legend: str) -> str:
    return f"![{n}　{title}](figures_v2/{stem}.png)\n\n**{n}　{title}。** {legend}"


def i0(x) -> str:
    return f"{int(round(float(x))):,}"


# ----------------------------------------------------------------------------- numbers from the figures
def numbers(R: dict) -> dict:
    N: dict = {}
    man = manifest()
    genes = pd.read_parquet(PUB / "genes.parquet")
    dis = pd.read_parquet(PUB / "diseases.parquet")
    f1 = man["fig1"]["numbers"]
    N.update(gini_g=f1["gini_genes"], gini_d=f1["gini_diseases"], top10=f1["top10_gene_share"],
             rho=f1["spearman_go_rel"])
    layers = ["has_go", "has_seq", "has_expr", "has_gene_hpo", "has_ppi", "has_rel"]
    N["n_genes"] = len(genes)
    N["layer_counts"] = {k: int(genes[k].sum()) for k in layers}
    N["all6"] = int(genes[layers].all(axis=1).sum())
    N["no_rel_but_go_seq_expr"] = int((genes["has_go"] & genes["has_seq"] & genes["has_expr"] & ~genes["has_rel"]).sum())
    N["rel_genes"] = int(genes["has_rel"].sum())
    nc = genes[genes["gtype"] == "ncrna"]
    N["ncrna_n"] = len(nc)
    N["ncrna_nogo_share"] = float((~nc["has_go"]).mean()) if len(nc) else None
    N["ncrna_noseq_share"] = float((~nc["has_seq"]).mean()) if len(nc) else None
    N["nogo_genes"] = int((~genes["has_go"]).sum())
    N["go_rich_rel_share"] = float(genes.loc[genes["n_go_pairs"] >= 100, "has_rel"].mean())
    N["go_rich_n"] = int((genes["n_go_pairs"] >= 100).sum())
    N["go_poor_rel_share"] = float(genes.loc[genes["n_go_pairs"].between(1, 4), "has_rel"].mean())
    e = sd("Fig1_knowledge_landscape", "e")
    N["dis_bins"] = e.to_dict("records")
    N["dis_total"] = len(dis)
    N["dis_with_rel"] = int((dis["n_rel"] > 0).sum())
    c2 = sd("Fig2_method_overview", "c").iloc[0]
    N["rrf_cut_aware"], N["rrf_cut_plain"] = float(c2["rB_cut_semantic_missing_aware"]), float(c2["rB_cut_semantic_missing_plain"])
    N["rrf_n"] = int(c2["n_diseases"])
    # ---- Fig 3
    a3 = sd("Fig3_main_results", "a")
    N["h"] = a3.to_dict("records")
    c3 = sd("Fig3_main_results", "c")
    N["rk"] = {m: {int(k): float(v) for k, v in zip(s["K"], s["recall"])} for m, s in c3.groupby("method")}
    d3 = sd("Fig3_main_results", "d")
    one = d3[d3["vs"] == "channel_lgb_v6"]
    two = d3[d3["vs"] == "v71_direct_strict"]
    N["fs_pos"] = int((one["delta"] > 0).sum())
    N["fs_ci_pos"] = int((one["lo"] > 0).sum())
    N["fs_n"] = len(one)
    N["fs_one_range"] = (float(one["delta"].min()), float(one["delta"].max()))
    N["fs_two_range"] = (float(two["delta"].min()), float(two["delta"].max()))
    N["fs_two_pos"] = int((two["delta"] > 0).sum())
    fw = sd("Fig3_main_results", "f")
    N["wtl"] = fw.to_dict("records")
    N["cold"] = sd("Fig3_main_results", "b").to_dict("records")
    # ---- Fig 4
    a4 = sd("Fig4_budget_fusion_flow", "a")
    at500 = {}
    for s_, g_ in a4.groupby("series"):
        g_ = g_.sort_values("K")
        at500[s_] = float(np.interp(500, g_["K"], g_["coverage"]))
        N[f"K50_{s_}"] = int(g_.loc[g_["coverage"] >= 0.5, "K"].min()) if (g_["coverage"] >= 0.5).any() else None
    N["cov500"] = man["fig4"]["numbers"]["coverage_at_500"]
    N["flow"] = {k: man["fig4"]["numbers"][k] for k in ("flow_up", "flow_down", "flow_up_cold", "flow_total")}
    N["miss"] = sd("Fig4_budget_fusion_flow", "c").to_dict("records")
    N["slope"] = sd("Fig4_budget_fusion_flow", "b").to_dict("records")
    d4 = sd("Fig4_budget_fusion_flow", "d").set_index("index")
    N["flow_matrix"] = d4
    # ---- Fig 5
    N["s5a"] = sd("Fig5_knowledge_strata", "a")
    N["s5b"] = sd("Fig5_knowledge_strata", "b")
    N["s5c"] = sd("Fig5_knowledge_strata", "c")
    N["s5d"] = sd("Fig5_knowledge_strata", "d")
    # ---- Fig 6
    N["dev_ab"] = pd.read_parquet(PUB / "dev_A.parquet")
    N["dev_c"] = pd.read_parquet(PUB / "dev_C.parquet")
    N["fi"] = sd("Fig6_second_stage_diagnostics", "d")
    fi = pd.read_parquet(PUB / "feature_importance.parquet")
    from .pubdata8 import feat_family
    fam_tot = fi.assign(fam=fi["feature"].map(feat_family)).groupby(["seed", "fold", "fam"])["gain_share"].sum()
    N["fi_fam"] = fam_tot.groupby("fam").mean().to_dict()
    cur = pd.read_parquet(PUB / "dev_C_curves.parquet")
    drops = []
    for (j, name), s_ in cur.groupby(["fold", "name"]):
        v = s_.sort_values("round")["val_recall10"].to_numpy()
        drops.append((v.max() - v[-1]))
    N["curve_drop_median"] = float(np.median(drops))
    N["curve_drop_max"] = float(np.max(drops))
    fold_drop = {}
    for j, s_j in cur.groupby("fold"):
        dj = []
        for _, s_ in s_j.groupby("name"):
            v = s_.sort_values("round")["val_recall10"].to_numpy()
            dj.append(v.max() - v[-1])
        fold_drop[int(j)] = float(np.median(dj))
    N["fold_drop"] = fold_drop
    # ---- Fig 7
    N["e5"] = sd("Fig7_go_mask_sequence", "d").set_index("arm")
    N["e6"] = sd("Fig7_go_mask_sequence", "e").set_index("arm")
    N["stu"] = sd("Fig7_go_mask_sequence", "b")
    # ---- Fig 8
    N["codes"] = sd("Fig8_evidence_sources", "a")
    N["del"] = sd("Fig8_evidence_sources", "c")
    N["dod"] = sd("Fig8_evidence_sources", "d")
    # ---- Fig 9
    N["ext"] = sd("Fig9_external_temporal_applications", "a")
    N["fields"] = sd("Fig9_external_temporal_applications", "b")
    N["hist"] = sd("Fig9_external_temporal_applications", "c")
    N["sup"] = sd("Fig9_external_temporal_applications", "d")
    N["expr"] = sd("Fig9_external_temporal_applications", "e")
    # ---- Fig 10 / S
    N["cases"] = man["fig10"].get("cases", [])
    N["case_rows"] = sd("Fig10_case_studies", "a")
    N["tsne"] = json.loads((PUB / "tsne_meta.json").read_text(encoding="utf-8"))
    N["tsne_auc"] = float(sd("FigS5_embedding", "b")["auc"].iloc[0])
    led = pd.read_parquet(PUB / "ledger.parquet")
    N["runs"] = int(len(led))
    N["timed_h"] = float(led["seconds"].fillna(0).sum() / 3600)
    N["gpu_peak"] = float(led["peak_gpu_allocated_mib"].max())
    N["rss_peak"] = float(led["peak_rss_mib"].max())
    N["budget_exact"] = json.loads((PUB / "budget_checks.json").read_text(encoding="utf-8"))["checks"]
    th = pd.read_parquet(PUB / "teacher_histories.parquet")
    b = th.groupby(["ctx", "controls", "seed"])["best_epoch"].first()
    N["teacher_best"] = (int(b.min()), int(b.max()))
    N["x"] = numbers_x()
    return N


def numbers_x() -> dict:
    """Numbers of the second-round (high-complexity) panels, all read from their source-data tables."""
    from .pubdata8 import MESH_CAT
    X: dict = {}
    nm = lambda c: MESH_CAT.get(c, c)
    # Fig 1 f/g
    f1 = sd("Fig1_knowledge_landscape", "f").sort_values("jaccard", ascending=False)
    X["jac_top"] = [(nm(r.a), nm(r.b), float(r.jaccard), int(r.shared_genes)) for r in f1.head(3).itertuples()]
    X["jac_low"] = [(nm(r.a), nm(r.b), float(r.jaccard)) for r in f1.tail(1).itertuples()][0]
    g1 = sd("Fig1_knowledge_landscape", "g")
    hs = g1.sort_values("hpo_share")
    X["hpo_lo"], X["hpo_hi"] = (nm(hs.iloc[0]["code"]), float(hs.iloc[0]["hpo_share"])), \
        (nm(hs.iloc[-1]["code"]), float(hs.iloc[-1]["hpo_share"]))
    gm = g1.sort_values("gene_go_median")
    X["go_lo"], X["go_hi"] = (nm(gm.iloc[0]["code"]), float(gm.iloc[0]["gene_go_median"])), \
        (nm(gm.iloc[-1]["code"]), float(gm.iloc[-1]["gene_go_median"]))
    X["n_cat"] = len(g1)
    # Fig 2 e (UpSet)
    u = sd("Fig2_method_overview", "e")
    u["combo"] = u["combo"].astype(str).str.zfill(7)        # CSV round trip drops leading zeros
    X["up_n_combo"] = len(u)
    X["up_all7"] = int(u.loc[u["combo"] == "1111111", "n_diseases"].sum())
    X["up_total"] = int(u["n_diseases"].sum())
    X["up_pos"] = int((u["ci_low"] > 0).sum())
    X["up_rows"] = u.to_dict("records")
    # Fig 3 g/h
    g3 = sd("Fig3_main_results", "g").set_index("method")
    X["unl"] = g3["unlisted_share"].to_dict()
    X["med_best"] = g3["median_best_rank_listed"].to_dict()
    h3 = sd("Fig3_main_results", "h")
    X["uniq"] = h3[h3["b"] == "(unique)"].set_index("a")["shared_top10_hits"].to_dict()
    pair = h3[h3["b"] != "(unique)"]
    X["shared"] = {(r.a, r.b): int(r.shared_top10_hits) for r in pair.itertuples()}
    qr = pd.read_parquet(PUB / "query_ranks.parquet", columns=["seed", "method", "rank"])
    q42 = qr[qr["seed"] == 42]
    X["hits_total"] = q42[(q42["rank"] > 0) & (q42["rank"] <= 10)].groupby("method").size().to_dict()
    # Fig 4 f/g
    X["quad"] = sd("Fig4_budget_fusion_flow", "f").set_index("region")["share"].to_dict()
    X["quad_n"] = int(sd("Fig4_budget_fusion_flow", "f")["n_semantic_available_positives"].iloc[0])
    X["resc"] = sd("Fig4_budget_fusion_flow", "g").set_index("group").to_dict("index")
    # Fig 5 e
    e5 = sd("Fig5_knowledge_strata", "e")
    X["lift"] = e5.assign(name=e5["code"].map(nm)).to_dict("records")
    # Fig 6 e-h (SHAP)
    X["shap_top"] = sd("Fig6_second_stage_diagnostics", "e").head(4).to_dict("records")
    X["shap_fam"] = sd("Fig6_second_stage_diagnostics", "h").set_index("stratum").to_dict("index")
    X["shap_meta"] = json.loads((PUB / "shap_f0_s42_meta.json").read_text(encoding="utf-8"))
    df_ = sd("Fig6_second_stage_diagnostics", "f").sort_values("x_bin_median")     # x = RRF rank
    pos_, neg_ = df_[df_["mean_shap"] > 0], df_[df_["mean_shap"] < 0]
    X["dep_f"] = {"first_rank": float(df_["x_bin_median"].iloc[0]), "first": float(df_["mean_shap"].iloc[0]),
                  "last_pos_rank": float(pos_["x_bin_median"].max()),
                  "last_pos": float(pos_.loc[pos_["x_bin_median"].idxmax(), "mean_shap"]),
                  "first_neg_rank": float(neg_["x_bin_median"].min()),
                  "tail": float(df_[df_["x_bin_median"] > 400]["mean_shap"].mean())}
    dg_ = sd("Fig6_second_stage_diagnostics", "g").sort_values("x_bin_median")     # x = s_F
    lastneg = dg_[dg_["mean_shap"] < 0]["x_bin_median"].max()
    after = dg_[dg_["x_bin_median"] > lastneg]
    X["dep_g"] = {"min": float(dg_["mean_shap"].min()), "zero_lo": float(lastneg),
                  "zero_hi": float(after[after["mean_shap"] > 0.02]["x_bin_median"].min()),
                  "top_x": float(dg_["x_bin_median"].iloc[-1]), "top": float(dg_["mean_shap"].iloc[-1])}
    # Fig 7 f-h
    X["e5r"] = sd("Fig7_go_mask_sequence", "f").set_index("arm").to_dict("index")
    X["e5p"] = sd("Fig7_go_mask_sequence", "h").set_index("arm").to_dict("index")
    fl = sd("Fig7_go_mask_sequence", "g")
    mn = fl[(fl["from_arm"] == "main") & (fl["to_arm"] == "e5_none")]
    X["flow_main_top10"] = int(mn[mn["from_class"] == "前 10"]["n"].sum())
    X["flow_main_top10_kept"] = int(mn[(mn["from_class"] == "前 10") & (mn["to_class"] == "前 10")]["n"].sum())
    X["flow_main_listed_lost"] = int(mn[(mn["from_class"] != "未入选") & (mn["to_class"] == "未入选")]["n"].sum())
    X["flow_main_listed"] = int(mn[mn["from_class"] != "未入选"]["n"].sum())
    # Fig 8 e
    e8 = sd("Fig8_evidence_sources", "e")
    cat8 = e8[e8["code"] != "ALL"]
    X["dod_iea_neg"] = int((cat8[cat8["arm"] == "e4_rescore_iea_only"]["dod_x100"] < 0).sum())
    X["dod_ncat"] = int(cat8["code"].nunique())
    ret = cat8[~cat8["arm"].str.startswith("e4_rescore")]
    X["dod_ret_range"] = (float(ret["dod_x100"].min()), float(ret["dod_x100"].max()))
    X["dod_ret_small"] = float((ret["dod_x100"].abs() <= 2.0).mean())
    pv = cat8.pivot_table(index="code", columns="arm", values="dod_x100")
    X["dod_sign_agree"] = [float((np.sign(pv[a]) == np.sign(pv[f"{a}_random"])).mean())
                           for a in ("e4_layer1", "e4_layer2", "e4_blocks")]
    # Fig 9 f/g
    X["hp"] = sd("Fig9_external_temporal_applications", "f").set_index("task").to_dict("index")
    g9 = sd("Fig9_external_temporal_applications", "g")
    X["sup_pairs"] = {int(k): {"up": int((s_["joint_v8"] > s_["joint_v6"]).sum()),
                               "down": int((s_["joint_v8"] < s_["joint_v6"]).sum()),
                               "same": int((s_["joint_v8"] == s_["joint_v6"]).sum())} for k, s_ in g9.groupby("k")}
    # Fig 10 b (networks)
    nets = []
    for k in (1, 2, 3):
        nd = sd("Fig10_case_studies", f"b_{k}_nodes")
        ed = sd("Fig10_case_studies", f"b_{k}_edges")
        dep = set(nd[nd["group"].isin(["dep", "both"])]["gene_row"])
        ch = set(nd[nd["group"].isin(["ch", "both"])]["gene_row"])
        phys = ed[ed["kind"] == "physical"]
        inside = lambda S: int(sum(1 for r in phys.itertuples() if r.a in S and r.b in S))
        nets.append({"disease_id": nd["disease_id"].iloc[0], "phys_dep": inside(dep), "phys_ch": inside(ch),
                     "go_edges": int((ed["kind"] == "go").sum()), "nodes": len(nd),
                     "shared": nd[nd["group"] == "both"]["symbol"].tolist()})
    X["nets"] = nets
    # Fig S3 b/c
    s3b = sd("FigS3_fold_seed_stability", "b")
    ch3 = s3b[(s3b["comparator"] == "channel_lgb_v6") & (s3b["fold"].astype(str) != "全部")]
    X["s3_better"] = (float(ch3["better"].min()), float(ch3["better"].max()))
    X["s3_worse"] = (float(ch3["worse"].min()), float(ch3["worse"].max()))
    X["s3_ci_pos"] = int((ch3["ci_low"] > 0).sum())
    X["s3c"] = sd("FigS3_fold_seed_stability", "c").to_dict("records")
    return X


# ----------------------------------------------------------------------------- abstract / intro
def abstract(R: dict, N: dict) -> str:
    dep, ch, v71 = method_row(R, "deployed"), method_row(R, "channel_lgb_v6"), method_row(R, "v71_direct_strict")
    h = g(R, "main", "hierarchy", default={}) or {}
    v = g(R, "main", "verdict", default={}) or {}
    s5a = N["s5a"]
    cold = s5a[s5a["stratum"] == "冷"].set_index("method")["hit10"]
    rich = N["s5b"][N["s5b"]["stratum"] == "≥100"].set_index("method")["hit10"]
    zero = N["s5b"][N["s5b"]["stratum"] == "0"].set_index("method")["hit10"]
    return (
        f"**背景** 已知疾病—基因关联高度集中：本研究 19,861 个基因中，关联数最多的 10% 基因占全部已知关联的 {pc(N['top10'], 0)}"
        f"（基因 Gini 系数 {N['gini_g']:.2f}），功能注释、表型与序列的可用性也随研究程度同向变化。在这种知识覆盖不均衡下，"
        f"疾病关联冷启动的检索既要利用语义信息，又不能把“缺少某类输入”误当作“低匹配”。\n\n"
        f"**方法** 在五折外层划分、864 个开发家族疾病与 4,238 个测试疾病上执行事先登记的 V8.1 方案：按角色生成信息边界；"
        f"GO 注释按 17 列 GAF 审计并设三层删除与匹配随机对照；以复现的 V7.1 核心为语义教师，功能语义分数 $s_F$ 选检查点，"
        f"疾病长文本分段编码；以缺失感知的倒数名次融合生成严格 500 候选（语义缺失时精确退回检索名次）；在整链袋外特征上比较"
        f"门控冷平衡纠偏器与明示缺失值的直接 LightGBM（候选 A），按登记规则逐个外层上下文部署；序列学生只在功能注释缺失时补位。"
        f"全部选择早于任何测试评价登记；另做一组明确标注为事后描述的分层、预算与案例分析。\n\n"
        f"**结果** 部署模型（五个上下文均为候选 A）的端到端 Recall@10 为 {f(dep.get('recall10'))}，仅检索通道 LightGBM 在其自身 V6 500 名单上为 "
        f"{f(ch.get('recall10'))}，差值 {ci(h.get('h1_vs_channel_lgb_v6'))}，15 个外层 × 种子组合中 {N['fs_pos']} 个为正；"
        f"与同一严格 500 上重训的 V7.1 直接融合（{f(v71.get('recall10'))}）相比为 {ci(h.get('h2_vs_v71_direct_strict'), 4)}，"
        f"{'满足' if v.get('h2_success') else '未满足'}登记的 −0.005 非劣界。严格 500 使测试阳性进入候选的比例由 V6 顺序的 "
        f"{pc(N['cov500']['rB'][0])} 提高到 {pc(N['cov500']['r_aware'][0])}；与普通 RRF 相比，缺失感知融合使非编码 RNA 与语义不可用阳性的"
        f"候选覆盖约增加一倍。门控纠偏器在开发与测试集上均未超过候选 A，序列学生降低了屏蔽基因的候选覆盖，均按规则作为阴性结果报告。"
        f"事后分层显示，收益集中在 GO 注释丰富的基因（≥100 个基因–术语对时前十恢复 {pc(rich.get('deployed'))} 对 {pc(rich.get('channel_lgb_v6'))}）"
        f"与有 HPO 描述的疾病；参考路径冷阳性的前十恢复由几乎为零（{pc(cold.get('channel_lgb_v6'), 2)}）提高到 {pc(cold.get('deployed'))}，"
        f"而没有 GO 的基因几乎得不到语义收益（{pc(zero.get('deployed'))}）。{_e4_sentence(R)}外部 618 个疾病再分析方向一致；"
        f"2023-12-31 截止的历史任务中未见相对强对照的收益（X2 {ci(g(R, 'historical_contrasts', 'x2_deployed_vs_channel_lgb_v6'))}）。\n\n"
        f"**结论** 在严格候选预算下，疾病—功能语义检索的收益得以保留并以缺失感知的方式进入最终排序；但它依赖可得的功能注释，"
        f"对注释缺失的基因帮助有限，新提出的冷平衡纠偏与序列补位在本协议下未取得收益。")


def _e4_sentence(R) -> str:
    from .manuscript8 import e4_sentence
    return e4_sentence(R)


INTRO = """## 1 引言

疾病基因检索的困难首先来自知识覆盖的不均衡。已知的疾病—基因关系集中在少数研究充分的基因上，新疾病或描述稀少的疾病缺少可直接迁移的参考关系；功能注释、表型注释和蛋白序列在基因之间的可用性也差异很大，而且与研究程度同向变化（图 1）。在这种条件下，一个检索系统的总体指标可能掩盖两类系统性偏差：缺少参考路径的“冷”基因被排序模型压低，以及缺少某一模态输入的候选被当作低匹配分数处理。

前一轮（V7.1）的多模态语义核心把疾病文本/表型与基因功能注释对齐到同一语义空间，在约 950 个候选的后融合中取得了主要增益，但也暴露了四个问题：GO 与关联标签可能共享证据、候选选择与最终排序偏好不一致、无效语义以 0 进入融合，以及序列在自然数据中主要是冗余信息。V8.1 不再扩大模态数量，而是把这些问题转化为可检验的设计：按角色生成信息边界并审计 GO 证据；在严格的 500 候选预算下，以缺失感知的倒数名次融合引入语义候选；在共享的基础排序上训练只作用于语义可用候选的冷平衡纠偏；把蛋白序列限定为功能注释缺失时的补位（图 2）。

本文的贡献与边界如下。第一，给出严格 500 预算下的端到端评价，把候选预算、二级模型与冷基因恢复分开报告，并把 V7.1 的约 950 预算结果作为桥接。第二，按事先登记的规则比较门控纠偏与明示缺失值的直接融合，并如实报告新纠偏器的阴性结果。第三，分清自然缺失、人工功能缺失与真正关联监督冷，在受控的 GO 屏蔽下检验序列能否补位。第四，以三层证据删除、匹配随机对照、历史截止与外部再分析检查结论对共享来源的依赖。第五，在登记推断之外，用明确标注为事后描述的分析刻画收益在“知识覆盖”各层中的分布，并给出按预先写定规则选择的案例。知识蒸馏、成对排序、倒数名次融合与缺失掩码都不是新概念；本文不把工程修复包装为独立的算法创新。
"""


# ----------------------------------------------------------------------------- 2.9-2.10 experiments
def recaption(block: str, old: str, new: str) -> str:
    return block.replace(old, new, 1)


def fold_table() -> str:
    ev = pd.read_parquet(paths8.RESULTS / "eval" / "eval_test_main.parquet")
    e = ev[(ev["method"] == "deployed") & (ev["seed"] == 42)]
    dis = pd.read_parquet(PUB / "diseases.parquet").set_index("disease_id")
    e = e.assign(fam=e["disease_id"].map(dis["family_group"]))
    rows = []
    for j, s_ in e.groupby("fold"):
        sel = json.loads((paths8.DEV / f"fold{j}" / "selection.json").read_text(encoding="utf-8"))
        tr = json.loads((paths8.RUNS / f"V8T__main__f{j}_outer__background__s42" / "result.json").read_text(encoding="utf-8"))
        rows.append([f"外层 {j}", f"$F_{j}$ / $F_{(j + 1) % 5}$", i0(len(s_)), i0(s_["n_query"].sum()), i0(s_["fam"].nunique()),
                     i0(tr["fit"]["n_reference"]), sel["w"]["w"], sel["ab"]["best_A"].replace("A_", "").replace("_", " "),
                     tr["fit"]["best_epoch"]])
    rows.append(["合计", "—", i0(len(e)), i0(e["n_query"].sum()), i0(e["fam"].nunique()), "—", "—", "—", "—"])
    return ("表 2　外层上下文的规模与登记选择（种子 42；测试、验证折按家族划分）\n\n" +
            table_md(["上下文", "测试 / 验证折", "测试疾病", "测试关系", "测试家族", "参考疾病", "$w_j$", "部署的候选 A",
                      "教师最佳轮次"], rows))


def design_table() -> str:
    rows = [
        ["E1", "严格 500 预算下的总体效应", "部署模型对仅检索通道 LGB（其自身 V6 500）与 V7.1 直接融合（同一严格 500 重训）；"
                                         "五折 × 三种子，逐病配对", "R@10（主终点）；$R^c$@10；层级①②③", "图 3；表 6"],
        ["E2", "门控冷平衡纠偏是否改善冷恢复", "β∈{0,1,2}（各折登记 λ）对部署模型；前十命中流量", "R@10、$R^c$@10、$R^{c\\to all}$@10",
         "图 6；表 7"],
        ["E3", "缺失处理与回退精确性", "$a=0$ 行 $s\\equiv B$；NaN 直接融合；非编码 RNA、无 GO 蛋白、语义不可用与零信号查询",
         "关系级入 500 与前十比例", "图 4c；表 7"],
        ["E4", "语义收益是否依赖共享证据", "三层删除 + 匹配随机、证据代码变体整链重训（种子 42）；去 IC、仅 IEA 重打分",
         "新旧方法差的变化（删减 − 完整）", "图 8；图 S7；表 8"],
        ["E5", "序列能否补位缺失的 GO", "$G_{\\mathrm{mask,test}}$ 删除 GO 后比较 7 种补位（含配对打乱与开放知识对照）",
         "屏蔽基因阳性入 500、前十；总体 ΔR@10", "图 7a–d、7f–h"],
        ["E6", "真正关联监督冷", "屏蔽基因同时撤出参考图、基因 HPO 与全部训练表；另设同时删除 GO", "同 E5", "图 7e"],
        ["E7", "候选预算的贡献", "固定 V6 500 / 共享严格 500 / 严格端到端 / 约 950 桥接；强对照同预算", "R@10 与预算归因差值",
         "图 4a–b"],
        ["E8", "外部与时间外推", "618 个疾病外部再分析（三个资格字段）；2023-12-31 截止的 X1/X2", "R@10；逐病配对差值", "图 9a–c、9f"],
        ["E9", "应用、审核与复现", "支持任务（533 病、1-shot 760 病）；四个表达队列；160 对盲审材料；第 0 折子链重算",
         "R@10；富集概率与秩检验 P；复现容差", "图 9d–e、9g；表 9"],
        ["事后", "收益在知识覆盖各层的分布与模型解释", "Recall@K、覆盖@K、按基因度/GO 数/疾病描述/MeSH 大类/来源可用性分层、"
         "名次联合分布与流量、逐病最佳名次、特征增益与 TreeSHAP、E4/E5 的类别与逐对展开、案例网络、嵌入",
         "描述性，带家族 bootstrap 区间，不作检验", "图 2e、3c、3g–i、4、5、6d–h、7f–h、8e、9f–g、10、S3、S5"],
    ]
    return "表 4　实验设计一览\n\n" + table_md(["编号", "问题", "设计", "指标", "图表"], rows)


def comparator_table() -> str:
    rows = [
        ["仅检索通道 LGB", "30 项 V6 通道特征", "其自身 V6 500；共享严格 500", "LightGBM lambdarank，$F_{j+1}$ 早停", "层级①对照"],
        ["基础排序器 B", "30 项 V6 通道（袋外 $B^{-b}$）", "严格 500", "同上，固定平移/尺度随模型保存", "纠偏器基准"],
        ["F7 固定融合", "7 个 V6 来源的固定权重", "V6 500；严格 500", "无训练", "规则基线"],
        ["V7.1 直接融合", "30 通道 + 14 项核心特征（缺失写 0）", "V6 500；严格 500", "LightGBM lambdarank，$F_{j+1}$ 早停", "层级②对照"],
        ["门控纠偏器", "49 维（通道 + 可用性 + 语义）", "严格 500", "自定义成对 PU 目标，15 叶，≤200 轮", "E2"],
        ["候选 A（部署）", "49 维，缺失以 NaN 明示", "严格 500", "lambdarank；叶数 {15,31,63} × 最小样本 {50,200}", "部署模型"],
        ["教师 $s_F$ / 核心 $r$ / 序列分数", "疾病文本与 HPO；基因 GO / 序列", "全背景（取前 500）", "V7.1 核心重训，$s_F$ 选检查点", "单独语义排名"],
        ["无学习 SapBERT", "未投影的 SapBERT 疾病—功能相似度", "全背景", "无训练", "语义消融"],
        ["线性双塔", "同教师输入，线性投影", "全背景", "对比学习，$F_{j+1}$ 选轮次", "语义消融"],
        ["KDGene", "知识图谱嵌入（实体/关系 1,000 维）", "严格 500", "≤100 轮，耐心 10，$F_{j+1}$ 冷验证选轮次", "外部方法"],
        ["SIGN 式", "上下文视图 0–2 阶传播 + 两层解码器", "严格 500", "AdamW（1e-3，wd 1e-4），30 轮，$F_{j+1}$ 选轮次", "外部方法"],
        ["ProtST-ESM-1b", "文本—蛋白零样本（只加载权重）", "全背景（蛋白编码）", "无训练；未执行远程代码", "开放知识对照"],
    ]
    return "表 5　对比方法\n\n" + table_md(["方法", "输入", "候选名单", "训练与检查点", "角色"], rows)


def hyper_table() -> str:
    tr = json.loads((paths8.RUNS / "V8T__main__f0_outer__background__s42" / "result.json").read_text(encoding="utf-8"))
    tc = tr["config"]["train"]
    bm = json.loads((paths8.DEV / "fold0" / "B_outer" / "base_meta.json").read_text(encoding="utf-8"))["params"]
    cm = json.loads((paths8.DEV / "fold0" / "C_b2_l0.1" / "corrector_meta.json").read_text(encoding="utf-8"))["config"]
    rows = [
        ["语义教师", f"秩 {tr['config']['core']['rank']}，全局门控，{tr['fit']['n_params']:,} 个参数；$\\lambda_A$={tc['lambda_A']}，"
                  f"τ={tc['tau']}；AdamW lr {tc['lr']}、wd {tc['weight_decay']}；批 {tc['batch']}、锚批 {tc['anchor_batch']}、"
                  f"锚预热 {tc['anchor_steps']} 步；≤{tc['max_epochs']} 轮，耐心 {tc['patience']}；梯度裁剪 {tc['grad_clip']}",
         "未标注对照集 {参考关系基因, 全背景} 开发集选择"],
        ["序列学生", "480→256（GELU）→128→$L_2$（156,032 参数；线性 61,568）；lr 1e-3，批 256，≤300 轮，耐心 20",
         "wd ∈ {1e-4, 1e-2}；开发验收"],
        ["疾病文本", "SapBERT（冻结）；每段 ≤126 词片，句界优先；池化前 [CLS] 按词片数加权；名称只写一次", "一级/二级符号掩蔽"],
        ["缺失感知 RRF", "$K_0$=60；恰好 500 个候选；GeneID 破并列", "$w\\in\\{0.3,0.5,0.7,0.9\\}$ 开发集选择"],
        ["基础排序器 B", f"lambdarank；{bm['num_leaves']} 叶，最小样本 {bm['min_child_samples']}，lr {bm['learning_rate']}，"
                        f"≤{bm['n_estimators']} 棵（$F_{{j+1}}$ 早停），行/列采样 {bm['subsample']}/{bm['colsample_bytree']}", "V6 预设"],
        ["门控纠偏器", f"自定义目标；{cm['num_leaves']} 叶，lr {cm['learning_rate']}，≤{cm['rounds']} 轮，最小叶样本 "
                      f"{cm['min_data_in_leaf']}，最小 Hessian {cm['min_sum_hessian_in_leaf']}；每病未标注 64 个",
         "β ∈ {0,1,2} × λ ∈ {0.03, 0.1}"],
        ["候选 A", "lambdarank；lr 0.05，≤1,500 棵（$F_{j+1}$ 早停），行/列采样 0.8/0.8，确定性", "叶数 {15,31,63} × 最小样本 {50,200}"],
        ["推断", "家族 bootstrap：登记对比 10,000 次（种子 20260923），描述性区间 1,000–2,000 次", "按疾病家族重抽样"],
    ]
    return "表 3　实现细节与超参数\n\n" + table_md(["组件", "设置", "选择范围"], rows)


def experiments(R: dict, N: dict) -> str:
    env = ("Python 3.12.10、PyTorch 2.9.1（CUDA 12.8）、LightGBM 4.7.0、NumPy 2.5.3、pandas 3.0.6、SciPy 1.18.1、Matplotlib 3.11.2；"
           "单块 NVIDIA GeForce RTX 5060 Ti，Windows 11 Pro")
    ex = N["budget_exact"]
    n_ex = sum(v["aware_top500_equal_saved_pool"] for v in ex.values())
    n_cmp = sum(v["compared"] for v in ex.values())
    ev = pd.read_parquet(paths8.RESULTS / "eval" / "eval_test_main.parquet")
    e = ev[(ev["method"] == "deployed") & (ev["seed"] == 42)]
    per_fold = e.groupby("fold").agg(n=("disease_id", "size"), q=("n_query", "sum"))
    return "\n\n".join([
        "### 2.9 实验设置与实现细节",
        "**数据与划分。** 项目背景为 19,861 个可映射基因（含非编码 RNA、假基因与缺序列、缺 GO、缺表达的基因；表 1）。阳性为 CTD "
        "marker/mechanism 直接证据关系（31,577 条，覆盖 5,758 个疾病）；therapeutic 关系只作分层报告。外层五折沿用既有按家族的划分："
        "外层 $j$ 以 $F_j$ 为测试、$F_{j+1}$ 为验证，其余三折为训练；864 个开发家族疾病与之分离，用于全部配置选择。"
        f"每个外层的测试集有 {per_fold['n'].min():,}–{per_fold['n'].max():,} 个疾病、{per_fold['q'].min():,}–{per_fold['q'].max():,} 条关系"
        f"（表 2），五折合计 {int(per_fold['n'].sum()):,} 个测试疾病、{int(per_fold['q'].sum()):,} 条关系、1,260 个疾病家族。"
        "训练集合再按家族分成三个内块，形成每折 3 个内上下文与 1 个外上下文，共 20 个拟合上下文；每个外上下文约 2,540 个参考疾病。",
        table1(R),
        fold_table(),
        "**评价协议。** 每个测试疾病的候选名单恰好 500 个基因，二级模型只对名单内基因打分；端到端 Recall@10 以疾病为单位计算后"
        "宏平均，查询阳性不补入名单，零覆盖疾病保留在分母中。三个种子（42/52/62）先在疾病内平均，从不取“任一种子命中”的并集。"
        "冷恢复 $R^c$@10 只在含参考路径冷查询的疾病上定义（1,669 个），另报冷贡献 $R^{c\\to all}$@10。亚组指标（非编码 RNA、无 GO "
        "蛋白、语义不可用、屏蔽基因、证据块关系）为关系级微平均。",
        "**统计推断。** 所有区间为按疾病家族重抽样的 bootstrap 95% 百分位区间：登记的对比用 10,000 次（种子 20260923），与登记数字"
        "逐位一致；事后描述性区间用 1,000–2,000 次，关系级比例以家族为簇对分子与分母同时重抽样。判定层级①→②→③为固定顺序的门控"
        "检验，②只在①成立后检验、③只在②成立后检验，因此不需要额外的多重比较校正；其余比较与全部事后分析只报告估计与区间，"
        "不作显著性判定。开发集的 5×864 次评价是相关重复，不当作独立样本。",
        "**实现细节与超参数。** 主要设置见表 3；全部配置在开发集或同一上下文的 $F_{j+1}$ 上选择，每个外层上下文独立选择，"
        "测试后不补网格。教师以同上下文 $F_{j+1}$ 上 $s_F$ 的全背景 Recall@10 选检查点，二级模型以 $F_{j+1}$ 的 Recall@10 早停。",
        hyper_table(),
        f"**计算环境与可复现性。** 全部计算在单机完成：{env}。账本共记录 {N['runs']:,} 次拟合或评价运行（全部完成、内容哈希核对通过），"
        f"有计时的运行合计约 {N['timed_h']:.1f} 小时；单次运行的峰值 GPU 显存 {N['gpu_peak'] / 1024:.2f} GiB、峰值内存 "
        f"{N['rss_peak'] / 1024:.1f} GiB。为作图而在完整背景上重算的缺失感知 RRF 与登记时保存的严格 500 名单逐病比较，"
        f"{n_ex:,}/{n_cmp:,} 个测试疾病完全一致。第 0 折种子 42 子链以全新缓存重算，候选名单与前十完全一致、部署分数最大差 "
        f"{g(R, 'reproduction', 'deployed_scores_maxabs', default=0):.0e}（独立虚拟环境被系统策略阻止，见 3.12）。"
        "图件以 Python 绘制，矢量 PDF/SVG 与 600 dpi PNG 同时输出，每个面板的作图数据以 CSV 随图提供；配色使用经色觉缺陷模拟"
        "（Machado 2009）验证的固定分类色，颜色始终对应同一方法。",
        "**事后解释与可视化方法。** 模型解释使用 LightGBM 内置的 TreeSHAP（`pred_contrib`）：对外层 0、种子 42 的部署模型，由保存的"
        "模型字符串与按登记流程重建的测试特征矩阵计算全部留出阳性与 4 万个按固定种子抽样的未标注候选的逐特征贡献，并核对"
        "“SHAP 之和 = 保存的排序分数”（最大差约 1e-7）。分布类面板用高斯核密度（带宽默认取 Silverman 经验规则，部分面板为固定带宽；名次在对数尺度上，"
        "有界时做边界反射校正），二维分布用六边形分箱与核密度等高线；层次聚类为欧氏距离的平均连接；弦图的带宽与共享计数成正比；网络布局按"
        "方法分组的固定弧形排列，边取自 STRING v12 物理/功能连接与 GO 术语集 Jaccard。以上都只用于描述与展示。",
        "### 2.10 定向实验与比较器",
        "表 4 汇总九组登记实验与一组事后描述分析。E1–E9 的设计、分母与判定规则在任何 V8.1 测试评价之前登记；事后分析"
        "（Recall@K、候选覆盖@K、知识覆盖分层、疾病大类、来源可用性、名次联合分布与命中流量、特征增益与 TreeSHAP、嵌入与案例网络）"
        "只使用已冻结的预测与保存的模型，不回流任何选择，也不作显著性判定。",
        design_table(),
        "比较器（表 5）包括 F7 固定融合、仅检索通道 LightGBM（在其自身 V6 500 与共享严格 500 上）、V7.1 直接后融合在两种名单上重训、"
        "学习前的 SapBERT 功能相似度、线性双塔、V7.1 核心 $r$、教师 $s_F$ 与教师自身序列分数的全背景单独排名、KDGene 与 SIGN 式方法"
        "（在同一严格 500 名单上以 $F_{j+1}$ 选检查点，五折），以及 ProtST-ESM-1b 的零样本文本—蛋白检索（只加载权重，自写读出，"
        "未执行远程代码）。支持任务比较等权 RWR、语义+RWR（V6 与 V8.1 语义，$\\omega$ 开发重选）与连续时间量子游走。",
        comparator_table(),
        "历史任务使用逐项核对日期与 SHA-256 的截止前资源：CTD 关系（2023-12-05 存档）、CTD 疾病词表（2022-01-24）、HPO 与"
        "基因—表型（2023-10-09）、DO（2023-11-30）、GO/GOA（2023-10-09）、HGNC（2023-10-01）、GTEx v8 与 STRING v12；"
        "基因背景、MANE 2025 蛋白序列、2026 年的等价组/家族与冻结的超参数仍晚于截止日，逐项披露（表 S6）。历史世界重建文本、"
        "HPO 空间、基因 HPO 排除、TF-IDF、本体与 GO 功能锚，并用登记的“无原生上下文”设置重训全部模型；X2 为截止时已有关系疾病的"
        "新关系（排序时移除旧基因，不是疾病冷启动），X1 为截止时无关联监督的疾病。",
    ])


# ----------------------------------------------------------------------------- results
LEG_STAT = "区间为按疾病家族重抽样的 bootstrap 95% 区间"


def _row(df: pd.DataFrame, **kw) -> dict:
    s_ = df
    for k, v in kw.items():
        s_ = s_[s_[k] == v]
    return s_.iloc[0].to_dict() if len(s_) else {}


def main_table(R: dict) -> str:
    rows = []
    for r in g(R, "main", "table", default=[]) or []:
        rows.append([r.get("label", r["method"]), r.get("seeds", ""), f(r.get("recall10")), f(r.get("recall20")),
                     f(r.get("recall50")), f(r.get("hit10")), f(r.get("coverage500")), f(r.get("cold_recall10")),
                     f(r.get("cold_to_all10"))])
    return ("表 6　同资源主比较（$F_j$ 测试，4,238 个疾病，种子先在疾病内平均）\n\n" +
            table_md(["方法", "种子", "R@10", "R@20", "R@50", "Hit@10", "覆盖@500", "$R^c$@10", "$R^{c\\to all}$@10"], rows))


def sec_landscape(N: dict) -> str:
    lc = N["layer_counts"]
    G = N["n_genes"]
    bins = {b["known_relations"]: b for b in N["dis_bins"]}
    b0, b1, b20 = bins.get("0", {}), bins.get("1", {}), bins.get(">20", {})
    txt = (
        f"已知疾病—基因关联在基因与疾病两侧都高度集中（图 1a）。{G:,} 个背景基因中只有 {N['rel_genes']:,} 个（{pc(N['rel_genes'] / G)}）"
        f"有至少一条已知关联；关联数最多的 10% 基因占全部已知关联的 {pc(N['top10'], 0)}，基因侧 Gini 系数为 {N['gini_g']:.2f}，"
        f"{N['dis_with_rel']:,} 个有关联疾病的 Gini 系数为 {N['gini_d']:.2f}。每个基因的关联数、每个疾病的关联数与每个基因的 GO "
        f"注释数都呈重尾分布（图 1b）。关联知识与注释知识同向集中：GO 基因–术语对数与已知关联数的 Spearman 相关为 {N['rho']:.2f}；"
        + (f"GO 对数 ≥100 的 {N['go_rich_n']:,} 个基因全部已有已知关联" if N["go_rich_rel_share"] >= 1 else
           f"GO 对数 ≥100 的基因中 {pc(N['go_rich_rel_share'])} 已有已知关联") +
        f"，只有 1–4 个 GO 对的基因中这一比例为 {pc(N['go_poor_rel_share'])}（图 1c）。\n\n"
        f"六个信息层的可用性差异同样明显（图 1d）：有 GO 注释的基因 {lc['has_go']:,} 个、有蛋白序列 {lc['has_seq']:,} 个、有组织表达 "
        f"{lc['has_expr']:,} 个、有基因–表型注释 {lc['has_gene_hpo']:,} 个、有物理互作 {lc['has_ppi']:,} 个、有已知疾病关联 "
        f"{lc['has_rel']:,} 个；六层齐全的只有 {N['all6']:,} 个（{pc(N['all6'] / G)}），而有 GO、序列与表达却没有任何已知关联的基因有 "
        f"{N['no_rel_but_go_seq_expr']:,} 个——它们正是疾病关联冷启动中只能靠语义信息找回的候选。{N['ncrna_n']:,} 个非编码 RNA 中 "
        f"{pc(N['ncrna_nogo_share'])} 没有 GO、{pc(N['ncrna_noseq_share'])} 没有蛋白序列，是缺失信息最集中的人群。"
        f"疾病侧也有同样的梯度（图 1e）：没有已知关联的 {i0(b0.get('n', 0))} 个疾病中 {pc(b0.get('name_only'))} 只有名称与同义词，"
        f"只有 1 条关联的疾病多来自 OMIM 条目、以 HPO 描述为主（仅 HPO {pc(b1.get('hpo_only'))}），而关联 >20 条的疾病几乎都有定义文本"
        f"（{pc((b20.get('def_and_hpo') or 0) + (b20.get('def_only') or 0))}）。")
    X = N["x"]
    jt = "；".join(f"{a}—{b}（{j:.2f}，共享 {s:,} 个基因）" for a, b, j, s in X["jac_top"])
    txt += (
        f"\n\n疾病大类之间共享大量关联基因（图 1f）：在关联基因最多的 12 个 MeSH 大类中，Jaccard 系数最高的三对是{jt}，最低的"
        f"{X['jac_low'][0]}—{X['jac_low'][1]}也有 {X['jac_low'][2]:.2f}，同一批研究充分的基因在多个大类中反复出现。各大类的知识覆盖指标"
        f"（图 1g，有已知关联的疾病 ≥40 个的 {X['n_cat']} 个大类，行列按相似性聚类）同样不均：有关联疾病中带 HPO 描述的比例从{X['hpo_lo'][0]}的 {pc(X['hpo_lo'][1], 0)} 到"
        f"{X['hpo_hi'][0]}的 {pc(X['hpo_hi'][1], 0)}，关联基因的 GO 基因–术语对中位数从{X['go_lo'][0]}的 {X['go_lo'][1]:.0f} 到"
        f"{X['go_hi'][0]}的 {X['go_hi'][1]:.0f}。")
    leg = (f"**a** 按关联数升序排列的累计比例曲线（Lorenz 曲线）；阴影为基因曲线与均等线之间的面积，圆点标出前 10% 基因所占份额。"
           f"**b** 三种计数的互补累计分布 $P(X\\geq k)$（双对数轴，仅计 ≥1 的对象）。**c** 按 GO 注释数分组后各组内已知关联数的分布"
           f"（列内比例，≥10% 的格子标注数值；顶部为每组基因数）；ρ 为全体基因的 Spearman 相关。**d** 六个信息层的 UpSet 图："
           f"上方为前 14 个交集的基因数（深色为蛋白编码），下方点阵标出交集组成，左侧为各层大小。**e** 按已知关联数分组的疾病描述完整度"
           f"（括号内为疾病数）。**f** 关联基因最多的 12 个 MeSH 大类之间共享关联基因的弦图：弧长 ∝ 大类内关联基因数，带宽 ∝ 两类共享的"
           f"基因数，弧色为该类关联基因 GO 基因–术语对的中位数（蓝色阶），蓝色带为 Jaccard 最高的 6 对。**g** 各大类（有已知关联的疾病 ≥40 个）"
           f"的九项知识覆盖指标——疾病侧：疾病数、每病关联数中位、有定义与有 HPO 的比例、文本词片中位；基因侧（该类关联基因）：GO 基因–术语对"
           f"中位、已知关联数中位、有物理互作与无 GO 的比例。颜色为列内 z 分数（红低蓝高），格内为原始值；左侧与上方为平均连接的层次聚类树。"
           f"n = {G:,} 个基因，{N['dis_total']:,} 个疾病。")
    return "\n\n".join(["### 3.1 知识覆盖不均衡的数据全景", txt,
                        fig("Fig1_knowledge_landscape", "图 1", "关联知识与信息层可用性的不均衡", leg)])


def sec_method(R: dict, N: dict) -> str:
    txt2 = (f"缺失感知融合的效果可以直接从入选边界读出（图 2c）。在 $w=0.5$ 的四个外层上下文中，取 {N['rrf_n']:,} 个测试疾病第 500 个"
            f"候选分数的中位数作为边界：语义可用的基因只要检索名次与语义名次的组合落在曲线左下方即入选；语义缺失的基因在缺失感知融合下"
            f"只需 V6 名次 ≤{N['rrf_cut_aware']:,.0f}，而在普通 RRF 下必须 ≤{N['rrf_cut_plain']:,.0f}——普通 RRF 等于把缺少语义的候选额外"
            f"降一级。三类“冷”分开定义并分表报告（图 2d）：参考路径冷是自然状态，GO 输入缺失（E5）与真正关联监督冷（E6）是受控操作。")
    X = N["x"]
    ur = {r["combo"] if isinstance(r["combo"], str) else f"{int(r['combo']):07d}": r for r in X["up_rows"]}
    noh = ur.get("0111000", {})
    ot = ur.get("0011000", {})
    txt2 += (
        f"\n\n测试疾病上的 V6 来源可用性（图 2e）说明缺失是结构性的：{X['up_total']:,} 个测试疾病只出现 {X['up_n_combo']} 种可用性组合，"
        f"七个来源全部可用的 {X['up_all7']:,} 个（{pc(X['up_all7'] / X['up_total'], 0)}）；没有 HPO 的疾病同时失去表型、基因 HPO、表型 RWR "
        f"与兼容四个来源，其中只剩层级、本体与文本的 {i0(noh.get('n_diseases', 0))} 个疾病上部署模型相对仅通道 LightGBM 的差值为 "
        f"{noh.get('delta', 0):+.3f}（{noh.get('ci_low', 0):+.3f} 至 {noh.get('ci_high', 0):+.3f}），只剩本体与文本的 "
        f"{i0(ot.get('n_diseases', 0))} 个为 {ot.get('delta', 0):+.3f}（{ot.get('ci_low', 0):+.3f} 至 {ot.get('ci_high', 0):+.3f}）；"
        f"{X['up_pos']} 种组合的区间下界大于 0（事后描述）。")
    leg = ("**a** 定稿流程。输入按角色生成信息边界后，经 V6 七个检索来源得到完整背景顺序 $r_B$，经语义教师得到功能语义分数 $s_F$ 与"
           "语义名次 $r_S$；缺失感知 RRF 生成恰好 500 个候选，其上训练基础排序器 B、门控纠偏器与候选 A，逐个外层上下文按登记的 A/B "
           "规则部署。虚线框为阴性结果（序列学生未通过开发验收，自然主链中关闭）；虚线箭头为可关闭的表达特征。**b** 外层五折："
           "深色为测试折、中灰为验证折、浅色为训练折，训练折按家族分为三个内块 $I_{jb}$；右侧为与五折分离的开发、外部与历史集合。"
           "**c** $w=0.5$ 时缺失感知分数 $R$ 随 $r_B$、$r_S$ 的变化（对数色标）；黑线为测试疾病第 500 个候选分数的中位数边界，"
           "白线为四分位边界；下方色条为语义缺失（$m=0$）时的分数，竖线分别为普通 RRF 与缺失感知 RRF 的入选截断。"
           "**d** 三类“冷”在各信息项上的状态（实心：保留；空心：撤出或缺失）。**e** 测试疾病（5 折合计）的 V6 七个来源可用性 UpSet 图："
           "上方为每种可用性组合的疾病数，中间点阵标出组合内可用的来源，左侧为每个来源可用的疾病数；下方为该组合内部署模型相对仅通道 LGB 的"
           "逐病 Recall@10 差值（三种子平均；圆点与竖线为估计值与家族 bootstrap 95% 区间；蓝：区间 > 0，灰：区间含 0；事后描述）。")
    return "\n\n".join(["### 3.2 方法实现与输入核查", R.get("_txt_inputs", ""), txt2,
                        fig("Fig2_method_overview", "图 2", "V8.1 定稿方法与评价框架", leg)])


def sec_dev(R: dict, N: dict) -> str:
    tb = N["teacher_best"]
    return "\n\n".join([
        "### 3.3 开发选择",
        R.get("_txt_dev", ""),
        f"语义教师在各上下文（含内上下文与三个种子）的最佳轮次为 {tb[0]}–{tb[1]}（图 S1a），对照集的选择见图 S1b；"
        f"RRF 权重、候选 A 的六个配置与纠偏器—A 的开发差距见图 S2 与表 S1。",
    ])


def sec_e1(R: dict, N: dict) -> str:
    dep, ch, v71 = method_row(R, "deployed"), method_row(R, "channel_lgb_v6"), method_row(R, "v71_direct_strict")
    h = g(R, "main", "hierarchy", default={}) or {}
    v = g(R, "main", "verdict", default={}) or {}
    ps = g(R, "main", "deployed_per_seed", default={}) or {}
    rk = N["rk"]
    ks = [1, 5, 20, 50, 100]
    rk_txt = "、".join(f"K={k} 时 {rk['deployed'][k]:.3f} 对 {rk['channel_lgb_v6'][k]:.3f}" for k in ks)
    w_all = _row(pd.DataFrame(N["wtl"]), stratum="全部")
    w_gt5 = _row(pd.DataFrame(N["wtl"]), stratum="阳性数 >5")
    w_1 = _row(pd.DataFrame(N["wtl"]), stratum="阳性数 1")
    cold = {r["method"]: r for r in N["cold"]}
    txt = (
        f"部署模型的端到端 Recall@10 为 {f(dep.get('recall10'))}（种子 42/52/62 分别为 "
        f"{'、'.join(f'{float(x):.3f}' for x in ps.values())}），冷恢复 $R^c$@10 为 {f(dep.get('cold_recall10'))}（表 6）。"
        f"登记层级①：相对仅检索通道 LightGBM 在其自身 V6 500 名单上（{f(ch.get('recall10'))}），差值为 "
        f"{ci(h.get('h1_vs_channel_lgb_v6'), 4)}，{'下界大于 0，成立' if v.get('h1_success') else '未成立'}；10,000 次家族 bootstrap 的"
        f"分布整体位于 0 的右侧（图 3a）。层级②：相对在同一严格 500 上重训的 V7.1 直接融合（{f(v71.get('recall10'))}），差值为 "
        f"{ci(h.get('h2_vs_v71_direct_strict'), 4)}，{'下界高于 −0.005，非劣成立' if v.get('h2_success') else '下界低于 −0.005，非劣未确立'}；"
        + ("层级③随之检验：" + ci(h.get("h3_cold_vs_v71_direct_strict"), 4) + "。" if v.get("h3_tested") else
           "按登记，层级③不作推断，只描述冷恢复差值 " + ci(h.get("h3_cold_vs_v71_direct_strict"), 4) + "。") +
        f"\n\n结果在外层与种子之间稳定（图 3d）：15 个外层 × 种子组合中，相对仅通道 LightGBM 的差值 {N['fs_pos']} 个为正"
        f"（{N['fs_ci_pos']} 个的区间不含 0），范围 {N['fs_one_range'][0]:+.3f} 至 {N['fs_one_range'][1]:+.3f}；相对 V7.1 直接融合的差值"
        f"在 {N['fs_two_range'][0]:+.3f} 至 {N['fs_two_range'][1]:+.3f} 之间摆动（{N['fs_two_pos']} 个为正），与“两者相当”一致。"
        f"Recall@K 曲线在全部截断位置上保持同样的排序（图 3c）：部署模型对仅通道 LightGBM 为 {rk_txt}；部署模型与 V7.1 直接融合"
        f"的曲线几乎重合，全背景的教师 $s_F$ 单独排名在 K≈10 以上接近仅通道 LightGBM。逐疾病比较（图 3f），部署模型更好的疾病占 "
        f"{pc(w_all.get('deployed_better'), 0)}、更差的占 {pc(w_all.get('channel_better'), 0)}、其余 {pc(w_all.get('tie'), 0)} 持平；"
        f"持平主要来自只有 1 个阳性的疾病（持平 {pc(w_1.get('tie'), 0)}），阳性数 >5 的疾病中改善与变差分别为 "
        f"{pc(w_gt5.get('deployed_better'), 0)} 与 {pc(w_gt5.get('channel_better'), 0)}。"
        f"\n\n参考路径冷目标几乎只能由语义恢复（图 3b）：在 {i0(cold['deployed']['n_cold_diseases'])} 个含冷查询的疾病上，"
        f"仅检索通道的方法 $R^c$@10 只有 {cold['channel_lgb_v6']['Rc10']:.3f}（基础排序器 {cold['B']['Rc10']:.3f}），部署模型为 "
        f"{cold['deployed']['Rc10']:.3f}，V7.1 直接融合为 {cold['v71_direct_strict']['Rc10']:.3f}；全背景的教师 $s_F$ 单独排名的冷恢复"
        f"最高（{cold['standalone_sF']['Rc10']:.3f}），但它的总体 Recall@10 只有 {f(method_row(R, 'standalone_sF').get('recall10'))}。")
    X = N["x"]
    u, mb, uq, sh = X["unl"], X["med_best"], X["uniq"], X["shared"]
    txt += (
        f"\n\n以每个疾病最靠前的阳性名次看（图 3g，种子 42，事后描述），共享严格 500 的三种方法有 {pc(u['deployed'], 0)} 的疾病没有任何阳性"
        f"进入名单，仅通道 LightGBM 的 V6 500 为 {pc(u['channel_lgb_v6'], 0)}；入选疾病中最佳名次的中位数部署模型为 {mb['deployed']:.0f}、"
        f"V7.1 直接融合 {mb['v71_direct_strict']:.0f}、仅通道 LightGBM {mb['channel_lgb_v6']:.0f}，只用 30 个检索通道特征的基础"
        f"排序器 B 为 {mb['B']:.0f}——在同一严格 500 上，加入语义与可用性特征的二级模型把最佳阳性的中位名次提前了一半以上。五种方法的前十命中高度重叠（图 3h）：部署模型与 V7.1 直接融合共享 "
        f"{sh.get(('deployed', 'v71_direct_strict'), 0):,} 个命中，与仅通道 LightGBM 共享 {sh.get(('deployed', 'channel_lgb_v6'), 0):,} 个；"
        f"部署模型独有 {uq.get('deployed', 0):,} 个、仅通道 LightGBM 独有 {uq.get('channel_lgb_v6', 0):,} 个，全背景教师 $s_F$ 单独独有 "
        f"{uq.get('standalone_sF', 0):,} 个——教师 $s_F$ 单独的 {X['hits_total'].get('standalone_sF', 0):,} 个前十命中中，"
        f"{sh.get(('deployed', 'standalone_sF'), 0):,} 个（{pc(sh.get(('deployed', 'standalone_sF'), 0) / max(X['hits_total'].get('standalone_sF', 1), 1), 0)}）"
        f"也在部署模型的前十中，其余多为其他方法都没有找到的阳性。七项指标的雷达图（图 3i）"
        f"概括了这一格局：部署模型与 V7.1 直接融合几乎重合，仅通道 LightGBM 在冷恢复上接近零，教师 $s_F$ 单独只在冷恢复上领先。")
    leg = (f"**a** 事先登记的判定层级：半小提琴为 10,000 次家族 bootstrap 的差值分布，圆点与横线为估计值与 95% 区间，右侧为数值与判定；"
           f"浅红区为 −0.005 非劣界以外。**b** 冷恢复：实心为 $R^c$@10（只在含冷查询的疾病上定义），空心为 $R^{{c\\to all}}$@10（冷命中占"
           f"全部查询的比例）。**c** Recall@K 曲线（K = 1–100，对数轴），带为 1,000 次家族 bootstrap 的逐点 95% 置信带。**d** 15 个外层 × "
           f"种子组合中部署模型的差值（形状区分种子；黑色菱形为三种子合并的登记估计）。**e** 同资源主比较的全部指标（列内着色，右侧色块"
           f"为方法的固定颜色）。**f** 逐疾病的胜负比例（按疾病的三种子平均 Recall@10 比较），括号内为疾病数。**g** 每个疾病最靠前阳性"
           f"名次的云雨图（种子 42；上为核密度，中为箱线，下为抽样的逐病点；只画入选疾病，对数轴），右侧为没有阳性入选的疾病比例。"
           f"**h** 五种方法前十命中（疾病—基因对，种子 42）的重叠弦图：弧长 ∝ 该方法的前十命中数，带宽 ∝ 两种方法共享的命中数。"
           f"**i** 七项指标的雷达图（各轴除以四种方法中的最大值；MRR 为疾病内阳性倒数名次的平均）。g–i 为事后描述。n = 4,238 个测试疾病、"
           f"1,260 个家族；{LEG_STAT}。")
    return "\n\n".join(["### 3.4 严格 500 下的主要结果（E1）", txt, main_table(R),
                        fig("Fig3_main_results", "图 3", "严格 500 候选预算下的总体效应与稳定性", leg)])


def sec_budget(R: dict, N: dict) -> str:
    cv = N["cov500"]
    miss = pd.DataFrame(N["miss"])

    def mm(gr, li):
        r = _row(miss, group=gr, list=li)
        return pc(r.get("pool_share"))

    fl = N["flow"]
    hf = (R.get("hitflow", {}).get("flows", {}) or {}).get("channel_lgb_v6->deployed", {})
    bud = g(R, "budget", default={}) or {}
    M = N["flow_matrix"]
    try:
        out_to_top = int(M.loc["channel_未入选", "deployed_前 10"])
        top_to_out = int(M.loc["channel_前 10", "deployed_未入选"])
    except KeyError:
        out_to_top = top_to_out = None
    txt = (
        f"候选覆盖随预算 K 的曲线（图 4a，种子 42 完整背景重算）说明严格 500 的来源：K=500 时 V6 检索顺序覆盖 {pc(cv['rB'][0])} 的测试阳性，"
        f"语义顺序 {pc(cv['rS'][0])}，二者以缺失感知 RRF 融合后为 {pc(cv['r_aware'][0])}；覆盖一半测试阳性所需的预算由 V6 顺序的 "
        f"K≈{N.get('K50_rB') or '—'} 降到融合后的 K≈{N.get('K50_r_aware') or '—'}。普通 RRF 的总体曲线与缺失感知几乎重合"
        f"（K=500 时 {pc(cv['r_plain'][0])}），差别集中在缺少语义的阳性上（图 4c）：非编码 RNA 阳性进入 500 的比例在 V6 500、缺失感知与"
        f"普通 RRF 下分别为 {mm('非编码 RNA', 'V6 500（仅通道）')}、{mm('非编码 RNA', '缺失感知严格 500')} 与 {mm('非编码 RNA', '普通 RRF 严格 500')}，"
        f"语义不可用阳性为 {mm('语义不可用', 'V6 500（仅通道）')}、{mm('语义不可用', '缺失感知严格 500')} 与 {mm('语义不可用', '普通 RRF 严格 500')}；"
        f"缺失感知融合保住了这些候选，而普通 RRF 把它们的覆盖减半。"
        f"\n\n预算归因（图 4b）：在固定 V6 500 上，候选 A 相对仅通道 LightGBM 已高出 {ci(bud.get('v6pool_A_vs_channel'))}；同一候选 A "
        f"由 V6 500 换到严格 500 再增加 {ci(bud.get('strict_A_vs_v6pool_A'))}；由严格 500 扩到约 950 的桥接名单只再变化 "
        f"{ci(bud.get('bridge_A_vs_strict_A'))}，而同预算的通道排序器在三种名单上几乎不变。收益因此一部分来自二级模型对语义特征的利用，"
        f"一部分来自语义候选进入名单；把预算放宽到约 950 不再带来可检出的收益。"
        f"\n\n逐关系的名次流量（图 4d–e，种子 42，{fl['flow_total']:,} 条测试关系）：相对仅通道 LightGBM，部署模型使 {fl['flow_up']:,} 条"
        f"关系进入更好的名次区间（其中 {fl['flow_up_cold']:,} 条为参考路径冷），{fl['flow_down']:,} 条落入更差的区间；"
        + (f"原本未入选、在部署模型中进入前 10 的有 {out_to_top:,} 条，反向的有 {top_to_out:,} 条。" if out_to_top is not None else "")
        + (f"只看前十，保留 {hf.get('kept', 0):,} 个、失去 {hf.get('lost', 0):,} 个、新增 {hf.get('gained', 0):,} 个命中"
           f"（新增中 {hf.get('gained_cold', 0):,} 个为冷基因）。" if hf else ""))
    X = N["x"]
    q, rs = X["quad"], X["resc"]
    txt += (
        f"\n\n两条顺序的联合分布（图 4f，种子 42，{X['quad_n']:,} 个语义可用的测试阳性）说明语义候选的来源：{pc(q['均 ≤500'], 0)} 的阳性在"
        f"两条顺序中都进入前 500，{pc(q['仅检索 ≤500'], 0)} 只在检索顺序中、{pc(q['仅语义 ≤500'], 0)} 只在语义顺序中，{pc(q['均 >500'], 0)} "
        f"两者都没有；参考路径冷的阳性集中在检索名次约 $10^4$ 的区域，其中一部分的语义名次进入前 500。缺失感知融合的入选边界是一条双曲线：检索名次或语义名次单独足够靠前都能"
        f"入选。把 V6 500 换成严格 500 的净效果（图 4g）是 {pc(rs['全部阳性']['语义救回'])} 的阳性被语义救回、{pc(rs['全部阳性']['被挤出'])} "
        f"被挤出；参考路径冷的阳性中救回 {pc(rs['参考路径冷']['语义救回'])}、挤出只有 {pc(rs['参考路径冷']['被挤出'])}，而非编码 RNA 与没有 GO "
        f"的蛋白编码基因没有任何救回（语义不可用），只承担少量挤出（非编码 RNA {pc(rs['非编码 RNA']['被挤出'])}）。")
    leg = ("**a** 测试阳性进入前 K 的比例（疾病宏平均；K 为对数轴）：V6 检索顺序 $r_B$、语义顺序 $r_S$（语义不可用的基因视为未入选）、"
           "普通 RRF 与缺失感知 RRF（各外层的 $w_j$），带为家族 bootstrap 95% 置信带，竖线为 K=500 与约 950。**b** 预算归因："
           "同一方法在固定 V6 500、共享严格 500 与约 950 桥接名单上的 Recall@10（种子 42；竖线为 95% 区间），左下为三个归因差值。"
           "**c** 缺失信息阳性在三种名单下进入 500 的比例（关系级微平均，括号内为关系数；误差线为以家族为簇的 bootstrap 区间）。"
           "**d** 逐关系名次区间的流量（左：仅通道 LGB；右：部署模型）；蓝色为名次上升、橙色为下降、灰色为同一区间。**e** 同一批阳性"
           "在两种方法中名次的二维分布（对数色标，方框为对角线）。**f** 语义可用的测试阳性在 V6 检索名次 $r_B$ 与语义名次 $r_S$ 上的"
           "联合分布（种子 42；六边形分箱，对数色标；整数名次在 [r, r+1) 内均匀抖动后取对数，仅用于显示）：虚线为参考路径冷阳性的二维核密度"
           "等高线（三条线分别取网格密度值的第 50、80、95 百分位），黑线为 $w=0.5$ 时缺失感知 RRF 的中位入选边界，橙线与紫线为 $r_B=500$ 与 $r_S=500$；上方与右侧为边缘"
           "分布（深色为冷阳性），四角为各象限的阳性比例。**g** 由 V6 500 换为严格 500 时各类阳性的去向（种子 42，关系级）。"
           "f–g 为事后描述。")
    return "\n\n".join(["### 3.5 候选预算、缺失感知融合与命中流量（E3、E7）", txt,
                        recaption(table3(R), "表 3　冷目标与缺失亚组", "表 7　冷目标与缺失亚组"),
                        fig("Fig4_budget_fusion_flow", "图 4", "候选预算、缺失感知融合与名次流量", leg)])

def sec_strata(R: dict, N: dict) -> str:
    a, b, c, d = N["s5a"], N["s5b"], N["s5c"], N["s5d"]

    def hv(df, stratum, m):
        r = _row(df, stratum=stratum, method=m)
        return r.get("hit10")

    cold_d, cold_c = hv(a, "冷", "deployed"), hv(a, "冷", "channel_lgb_v6")
    cold_s = hv(a, "冷", "standalone_sF")
    z_d, z_c = hv(a, "0*", "deployed"), hv(a, "0*", "channel_lgb_v6")
    n_z = _row(a, stratum="0*", method="deployed").get("queries_per_seed")
    hi_d, hi_c, hi_v = hv(a, "≥32", "deployed"), hv(a, "≥32", "channel_lgb_v6"), hv(a, "≥32", "v71_direct_strict")
    n_cold = _row(a, stratum="冷", method="deployed").get("queries_per_seed")
    go0_d, go0_c = hv(b, "0", "deployed"), hv(b, "0", "channel_lgb_v6")
    go_hi_d, go_hi_c, go_hi_s = hv(b, "≥100", "deployed"), hv(b, "≥100", "channel_lgb_v6"), hv(b, "≥100", "standalone_sF")
    n_go0 = _row(b, stratum="0", method="deployed").get("queries_per_seed")
    cc = c.copy()
    hpo_q34 = cc[(cc["hpo"]) & (cc["token_quartile"] >= 3)]["delta"]
    nohpo = cc[~cc["hpo"]]["delta"]
    top = d.sort_values("delta", ascending=False)
    t3 = "；".join(f"{r['name']}（{r['delta']:+.3f}，{r['lo']:+.3f} 至 {r['hi']:+.3f}，n = {int(r['n']):,}）"
                   for _, r in top.head(3).iterrows())
    pos = int((d["lo"] > 0).sum())
    v71_in = int(((d["lo_v71"] <= 0) & (d["hi_v71"] >= 0)).sum())
    txt = (
        f"以下分析为事后描述，只使用冻结的测试预测，用来刻画收益落在“知识覆盖”的哪些层（图 5）。按基因在评价上下文中的参考关系数分层"
        f"（图 5a），参考路径冷的阳性（每个种子约 {i0(n_cold or 0)} 条）前十恢复率由仅通道 LightGBM 的几乎为零（{pc(cold_c, 2)}）"
        f"提高到 {pc(cold_d)}（全背景教师 $s_F$ 单独为 {pc(cold_s)}），绝对水平仍低；没有参考边但有基因 HPO 注释的少数阳性"
        f"（每个种子约 {i0(n_z or 0)} 条）则相反，仅通道 LightGBM 凭基因 HPO 通道达到 {pc(z_c)}，高于部署模型的 {pc(z_d)}。"
        f"其余各层的恢复率随参考关系数单调上升，参考关系 ≥32 的基因中部署模型、V7.1 直接融合与仅通道 LightGBM 分别为 "
        f"{pc(hi_d)}、{pc(hi_v)} 与 {pc(hi_c)}——绝对增益最大的反而是研究最充分的基因。按 GO 注释数分层（图 5b）"
        f"更直接地说明语义收益依赖功能注释：没有 GO 的基因（每个种子约 {i0(n_go0 or 0)} 条关系）两种方法都几乎恢复不了"
        f"（{pc(go0_d)} 对 {pc(go0_c)}），GO 对数 ≥100 的基因中部署模型为 {pc(go_hi_d)}、仅通道 LightGBM 为 {pc(go_hi_c)}、全背景教师"
        f" $s_F$ 单独为 {pc(go_hi_s)}。疾病侧（图 5c），有 HPO 描述且文本较长（第 3–4 四分位）的疾病增益为 "
        f"{hpo_q34.min():+.3f} 至 {hpo_q34.max():+.3f}，没有 HPO 的疾病只有 {nohpo.min():+.3f} 至 {nohpo.max():+.3f}。"
        f"\n\n按 MeSH 疾病大类（图 5d；一个疾病可属于多个大类，只列测试疾病 ≥60 的 {len(d)} 个大类），相对仅通道 LightGBM 的增益在 "
        f"{pos} 个大类中区间下界大于 0，增益最大的三个大类是{t3}；相对 V7.1 直接融合，{v71_in} 个大类的区间包含 0，"
        f"与总体上两者相当一致。"
        f"这些分层共同表明：V8.1 缓解的是“参考路径”上的不均衡，但语义通道本身建立在功能注释之上，注释覆盖的不均衡仍会传递到结果中。")
    lf = sorted(N["x"]["lift"], key=lambda r: -r["mean_log10_lift"])
    hi_, lo_ = lf[0], lf[-1]
    txt += (
        f"\n\n以逐病最佳阳性名次的对数提升看（图 5e，种子 42，只计至少一种方法有阳性入选的疾病），全部 {len(lf)} 个大类的平均提升都为正，"
        f"但分布很宽：{hi_['name']}最大（平均 {hi_['mean_log10_lift']:+.2f} 个数量级，{pc(hi_['better'], 0)} 的疾病提前、{pc(hi_['worse'], 0)} "
        f"推后），{lo_['name']}最小（{lo_['mean_log10_lift']:+.2f}，提前与推后各约 {pc(lo_['better'], 0)} 与 {pc(lo_['worse'], 0)}）；每个大类"
        f"都有相当比例的疾病最佳名次推后，与总体上“多数疾病持平或小幅改善、少数疾病变差”的逐病格局一致。")
    leg = (f"**a** 按基因在评价上下文中的参考关系数分层的关系级前十恢复率（三种子平均；“冷”为无参考边且无基因 HPO，“0*”为无参考边但"
           f"有基因 HPO；顶部 n 为每个种子的关系数）。**b** 按基因 GO 基因–术语对数分层。**c** 按疾病文本长度四分位与是否有 HPO 分层的"
           f"逐病 Recall@10 差值（部署减仅通道 LGB；格内为差值与疾病数，n<20 的格子不着色）。**d** MeSH 大类：左为四种方法的 Recall@10"
           f"（统一色标），中为部署模型相对仅通道 LGB（圆点）与 V7.1 直接融合（菱形）的差值，右为测试疾病数。**e** 各 MeSH 大类中逐病"
           f"最佳阳性名次的对数提升 $\\log_{{10}}$(仅通道最佳名次) − $\\log_{{10}}$(部署最佳名次) 的山脊图（种子 42；未入选记为 501；只计至少"
           f"一种方法有阳性入选的疾病）：填色为该类平均提升（红负蓝正），右侧为提前（↑）与推后（↓）的疾病比例和疾病数。"
           f"全部为事后描述，{LEG_STAT}。")
    return "\n\n".join(["### 3.6 收益在知识覆盖各层的分布（事后描述）", txt,
                        fig("Fig5_knowledge_strata", "图 5", "按基因与疾病的知识覆盖分层", leg)])


def sec_e2(R: dict, N: dict) -> str:
    dA, dC = N["dev_ab"], N["dev_c"]
    margins = [dA[dA["fold"] == j]["dev_recall10"].max() - dC[dC["fold"] == j]["dev_recall10"].max() for j in range(5)]
    bi = dC["best_iteration"]
    con = g(R, "e2", "contrasts", default={}) or {}
    fi = N["fi"].sort_values("mean", ascending=False)
    top = fi.head(3)
    fam = N["fi_fam"]
    fd = N["fold_drop"]
    worst = sorted(fd, key=lambda k: -fd[k])[:2]
    cold_hi = [j for j in range(5) if dC[dC["fold"] == j]["dev_cold_recall10"].max() >
               dA[dA["fold"] == j]["dev_cold_recall10"].min()]

    def lab_pct(lab, v):
        return (lab[:-1] + f"，{v:.1%}）") if lab.endswith("）") else f"{lab}（{v:.1%}）"

    txt = (
        f"开发阶段（图 6a–b），30 个纠偏配置（5 个上下文 × β 三档 × λ 两档）的验证曲线都在第 {int(bi.min())}–{int(bi.max())} 轮达到峰值，"
        f"随后持续下降：从峰值到第 200 轮的回落中位数为 {N['curve_drop_median']:.3f}、最大 {N['curve_drop_max']:.3f}，外层 "
        f"{worst[0]} 与 {worst[1]} 的回落最明显。在开发集上，最佳纠偏配置与最佳候选 A 的 Recall@10 差距为 "
        f"{min(margins):.3f}–{max(margins):.3f}，全部大于登记的 0.005，没有任何纠偏配置进入“最佳 A − 0.005”区间，五个上下文都按规则"
        f"部署候选 A；" + (f"在外层 {'、'.join(str(j) for j in cold_hi)} 中，最好的纠偏配置的开发冷恢复高于最差的 A 配置，但登记规则先看总体。"
                         if cold_hi else "纠偏器的开发冷恢复也没有高于任何 A 配置。") +
        f"测试集上（图 6c），β=0、1、2 的纠偏器相对部署模型的 Recall@10 差值分别为 "
        f"{ci(con.get('corrector_beta0', {}).get('recall10_vs_deployed'))}、{ci(con.get('corrector_beta1', {}).get('recall10_vs_deployed'))} 与 "
        f"{ci(con.get('corrector_beta2', {}).get('recall10_vs_deployed'))}，冷恢复差值的区间都包含 0；提高 β 没有改变冷恢复"
        f"（β2−β0：{ci(con.get('corrector_beta2_vs_beta0_cold'))}）。"
        f"\n\n部署的候选 A 在 15 个模型（5 个外层 × 3 个种子）中的特征增益（图 6d）说明它如何使用语义：增益最大的是"
        + "、".join(lab_pct(r["label"], r["mean"]) for _, r in top.iterrows()) +
        f"；按特征族合计，V6 检索通道占 {fam.get('V6 检索通道', 0):.0%}，候选融合名次占 {fam.get('候选融合', 0):.0%}，语义特征占 "
        f"{fam.get('语义', 0):.0%}，表达占 {fam.get('表达', 0):.0%}。缺失感知融合名次本身成为最有用的单一特征，说明二级模型主要是在"
        f"候选融合的顺序上做局部重排，而不是重新学习语义。")
    X = N["x"]
    st_, sf_, sm_ = X["shap_top"], X["shap_fam"], X["shap_meta"]
    top_txt = "、".join((r["label"][:-1] + f"；{r['mean_abs_shap']:.3f}）") if r["label"].endswith("）")
                        else f"{r['label']}（{r['mean_abs_shap']:.3f}）" for r in st_)
    ca, nc, ns = sf_["阳性·参考路径冷"], sf_["阳性·非冷"], sf_["阳性·语义缺失"]
    txt += (
        f"\n\nTreeSHAP 给出同一模型逐候选的归因（图 6e–h；外层 0、种子 42 的部署模型，{sm_['positives']:,} 个留出阳性与 "
        f"{sm_['negatives_sampled']:,} 个抽样的未标注候选；由保存的模型与重建的特征矩阵得到，逐候选的 SHAP 之和与保存的排序分数最大相差 "
        f"{sm_['reconstruction_maxabs_vs_saved_scores']:.1e}）。平均绝对 SHAP 最大的是{top_txt}（图 6e）。融合名次的贡献随名次近乎单调地下降"
        f"（分箱均值由名次约 {X['dep_f']['first_rank']:.0f} 时的 {X['dep_f']['first']:+.2f} 降到约 {X['dep_f']['last_pos_rank']:.0f} 时的 "
        f"{X['dep_f']['last_pos']:+.2f}），在名次约 {X['dep_f']['last_pos_rank']:.0f}–{X['dep_f']['first_neg_rank']:.0f} 之间由正转负并阶跃下降，"
        f"名次 400 以后约 {X['dep_f']['tail']:+.2f}（图 6f）；语义分数 $s_F$ 低于约 {X['dep_g']['zero_lo']:.2f} 时贡献为负（最低约 "
        f"{X['dep_g']['min']:+.2f}），在 {X['dep_g']['zero_lo']:.2f}–{X['dep_g']['zero_hi']:.2f} 附近接近 0，此后随 $s_F$ 迅速增大（约 "
        f"{X['dep_g']['top_x']:.2f} 时 {X['dep_g']['top']:+.2f}，图 6g）。按候选类别合计各"
        f"特征族的带符号 SHAP（图 6h），非冷阳性主要由 V6 检索通道（{nc['V6 检索通道']:+.2f}）推高，语义族贡献 {nc['语义']:+.2f}；参考路径冷的"
        f"阳性则相反，检索通道把它们拉低（{ca['V6 检索通道']:+.2f}），靠语义族（{ca['语义']:+.2f}）与融合名次（{ca['候选融合']:+.2f}）推高；"
        f"语义缺失的少数阳性（n = {i0(sf_['阳性·语义缺失']['n'])}）的语义族贡献为 {ns['语义']:+.2f}，即缺失本身被模型当作负面信号——这与"
        f"缺失感知融合“不因缺失额外降分”的设计意图不同，发生在二级模型而不是候选生成。以上为单一上下文的事后解释。")
    leg = ("**a** 开发阶段门控纠偏器在各外层 $F_{j+1}$ 上的验证 Recall@10 随提升轮次的变化（种子 42）；颜色深浅为 β，实线/点线为 λ，"
           "圆点为最佳轮次。**b** 开发集上各模型的 Recall@10 与冷恢复 $R^c$@10：蓝色为候选 A 的六个配置（黑圈为部署的配置），品红为纠偏器"
           "（形状为 β，实心/空心为 λ），黄色为基础排序器；浅蓝带为“最佳 A − 0.005 以内”的可部署区间。**c** 测试集上三个 β 的纠偏器相对部署"
           "模型的差值（三种子；各折使用登记的 λ），右图下方为 β 之间的冷恢复对比。**d** 部署的候选 A 的特征增益占比（均值 ± 标准差，"
           "n = 15 个模型），颜色为特征族；右侧为各特征族的合计。**e** 外层 0、种子 42 的部署模型在测试候选上的 TreeSHAP 蜂群图（全部留出"
           "阳性与 4 万个抽样的未标注候选；按平均绝对 SHAP 取前 14 个特征；点色为特征值在 2–98 百分位内的相对位置，灰色为缺失值）。"
           "**f** 融合名次（缺失感知 RRF 名次，对数轴）的 SHAP 依赖；**g** 语义分数 $s_F$ 的 SHAP 依赖：点色为 V6 检索名次 $r_B$"
           "（以 500 为中点的发散色标），空心圈为抽样的 700 个留出阳性，黑线与灰带为分位分箱的均值与 P10–P90。**h** 按候选类别合计的各特征族"
           f"带符号平均 SHAP（括号内为候选数）。e–h 为单一上下文的事后解释。{LEG_STAT}。")
    return "\n\n".join(["### 3.7 二级模型与门控纠偏器诊断（E2）", txt,
                        fig("Fig6_second_stage_diagnostics", "图 6", "二级模型、门控纠偏器的诊断与 SHAP 解释", leg)])


def sec_e56(R: dict, N: dict) -> str:
    e5, e6 = N["e5"], N["e6"]
    st = N["stu"]
    neg = int((st["gain"] < 0).sum())
    shuf_lower = int((st["shuffled_gain"] < st["gain"]).sum())
    top10_mlp = st[st["config"].str.startswith("mlp")]["top10_self"]
    top10_lin = st[st["config"].str.startswith("linear")]["top10_self"]

    def p_(arm, col):
        return pc(e5.loc[arm, col]) if arm in e5.index else "—"

    txt = (
        f"GO 屏蔽集合在 GO 术语数与物理 PPI 度的十分位层内按哈希抽取，屏蔽与未屏蔽基因的两项分布几乎重合（图 7a）。开发屏蔽臂上，"
        f"四种学生配置在 5 个外层上下文中的 {len(st)} 次验收" + (f"全部" if neg == len(st) else f"有 {neg} 次") +
        f"使屏蔽基因阳性进入 500 的比例下降（图 7b），长度层内打乱序列—基因配对后" + ("每一次都下降得更多；" if shuf_lower == len(st) else f"其中 {shuf_lower} 次下降得更多；") +
        f"学生输出在约 1.9 万个真实功能向量中找回自身的前十命中率为 {top10_mlp.min():.1%}–{top10_mlp.max():.1%}（两层）"
        f"与 {top10_lin.min():.1%}–{top10_lin.max():.1%}（线性），与真实向量的平均余弦明显高于两两平均余弦（图 7c），说明输出并非无信息，"
        f"但质量不足以替代观察到的 GO。"
        f"\n\n测试屏蔽臂（E5，图 7d）：同一批屏蔽基因在主模型（GO 未删）中进入 500 与前十的比例为 {p_('main', 'mask_pool')} 与 "
        f"{p_('main', 'mask_hit10')}；删除 GO 且不补位时降为 {p_('e5_none', 'mask_pool')} 与 {p_('e5_none', 'mask_hit10')}。两层学生、线性学生、"
        f"序列近邻转移与教师自身序列投影的候选覆盖分别为 {p_('e5_mlp', 'mask_pool')}、{p_('e5_linear', 'mask_pool')}、"
        f"{p_('e5_nn', 'mask_pool')} 与 {p_('e5_teacher_sS', 'mask_pool')}，都低于不补位；配对打乱后为 {p_('e5_mlp_shuffled', 'mask_pool')}。"
        f"ProtST 输入的学生（开放知识对照）为 {p_('e5_protst', 'mask_pool')}，同样没有超过不补位。补位不能提高候选覆盖的原因与方法设计"
        f"一致：缺失感知融合在语义缺失时保留检索分数，一个质量中等的补位语义名次反而会把融合分拉低。"
        f"\n\n真正关联监督冷（E6，图 7e）：撤出屏蔽基因的全部关联监督而保留 GO 时，进入 500 的比例为 {pc(e6.loc['e6_cold', 'mask_pool'])}、"
        f"前十 {pc(e6.loc['e6_cold', 'mask_hit10'])}；再删除 GO 后降为 {pc(e6.loc['e6_cold_gomask', 'mask_pool'])} 与 "
        f"{pc(e6.loc['e6_cold_gomask', 'mask_hit10'])}。在没有任何关联监督时，GO 语义仍能把约四分之一的阳性带入候选，但很少进入前十。")
    X = N["x"]
    pr = X["e5p"]
    mlp, prot, shf = pr["e5_mlp"], pr["e5_protst"], pr["e5_mlp_shuffled"]
    txt += (
        f"\n\n名次层面的配对比较（图 7f–h，种子 42，同一批 {i0(mlp['n'])} 个屏蔽基因阳性）给出同样的结论。删除 GO 后，主模型中前十的 "
        f"{X['flow_main_top10']:,} 个阳性只有 {X['flow_main_top10_kept']:,} 个仍在前十，原本入选的 {X['flow_main_listed']:,} 个中 "
        f"{X['flow_main_listed_lost']:,} 个跌出名单（图 7g）。与不补位相比，两层学生使 {mlp['better']:,} 个阳性名次上升、{mlp['worse']:,} 个下降，"
        f"其中由未入选变为入选的 {mlp['rescued_unlisted_to_listed']:,} 个、由入选变为未入选的 {mlp['lost_listed_to_unlisted']:,} 个（图 7h）；"
        f"每一种补位方式都是下降多于上升，ProtST 输入最接近持平（{prot['better']:,} 对 {prot['worse']:,}），配对打乱最差（{shf['better']:,} 对 "
        f"{shf['worse']:,}）。入选阳性的名次分布（图 7f）在各补位臂之间差别不大，差别主要在“是否入选”。")
    leg = ("**a** 未屏蔽、开发屏蔽（10%）与测试屏蔽（20%）基因的 GO 术语数与物理 PPI 度的累计分布（度为 0 的基因画在 0.5 处）。"
           "**b** 开发屏蔽臂上学生的验收：屏蔽基因阳性进入 500 的比例相对不补位的变化（实心，5 个外层上下文），空心为长度层内打乱配对的"
           "对照；浅蓝区为验收所需的正值区。**c** 屏蔽基因的学生输出质量：左为在约 1.9 万个真实功能向量中找回自身的前十命中率，右为与"
           "真实向量的余弦（实心）和两两平均余弦（空心）。**d** 测试屏蔽臂（种子 42）：各补位方式下屏蔽基因阳性进入 500 与前十的比例，以及"
           "总体 Recall@10 相对主模型的差值；竖线为不补位的水平。**e** 关联监督冷：主模型、删除 GO、撤出关联监督及两者同时时屏蔽基因阳性"
           "进入 500（浅）与前十（深）的比例。**f** 屏蔽基因阳性在严格 500 中名次的山脊图（种子 42，全部外层；只画入选者，对数轴，边界处"
           "反射校正的核密度；竖线为中位数），右侧为入选比例与前十比例。**g** 同一批阳性的名次类别在主模型 → 删 GO·无补位 → 删 GO·两层"
           "学生之间的流转（前 10 / 11–50 / 51–500 / 未入选；带宽 ∝ 阳性数）。**h** 各补位方式相对无补位的逐对名次变化：向右为名次上升、"
           f"向左为下降，深色窄条为其中跨越入选边界的阳性。f–h 为事后描述。{LEG_STAT}。")
    return "\n\n".join(["### 3.8 功能注释缺失、序列补位与关联监督冷（E5、E6）", txt,
                        fig("Fig7_go_mask_sequence", "图 7", "受控 GO 缺失、序列补位与关联监督冷", leg)])


def sec_e4(R: dict, N: dict) -> str:
    codes = N["codes"].copy()
    codes["share_no"] = codes["no_pmid"] / codes["total"]
    tot_no = codes["no_pmid"].sum() / codes["total"].sum()

    def cs(c):
        r = codes[codes["evidence_code"] == c]
        return (int(r["total"].iloc[0]), float(r["share_no"].iloc[0])) if len(r) else (0, 0.0)

    iea, tas, iba, ipi = cs("IEA"), cs("TAS"), cs("IBA"), cs("IPI")
    au = json.loads((paths8.AUDIT / "evidence_policy_audit_v8.json").read_text(encoding="utf-8"))
    l3 = au["layer3_evidence_blocks"]
    l2 = au["layer2_evaluation_conditioned"]
    dod = N["dod"]
    retrain = dod[dod["group"] != "rescore"]
    inc0 = int(((retrain["ci_low"] <= 0) & (retrain["ci_high"] >= 0)).sum())
    nm = _row(dod, arm="e4_no_mutant")
    iea_r = _row(dod, arm="e4_rescore_iea_only")
    txt = (
        f"生产规则保留的 845,871 条 GO 记录中 {pc(tot_no)} 没有 PMID（图 8a）：IEA（{iea[0]:,} 条）与 IBA（{iba[0]:,} 条）全部没有，"
        f"TAS（{tas[0]:,} 条）有 {pc(tas[1], 0)} 没有；数量最多的 IPI（{ipi[0]:,} 条）几乎全部可追溯。没有 PMID 的注释按未知出处处理，"
        f"不计为独立证据。PMID—关系二部图有 {l3['n_components']:,} 个连通块（图 8b），中位大小 {l3['component_size_quantiles']['0.5']:.0f}、"
        f"P99 为 {l3['component_size_quantiles']['0.99']:.0f}，另有一个含 {l3['giant_component_relations']:,} 条关系"
        f"（{pc(l3['giant_share'])}）的巨块；巨块不拆分、不抽取，随机抽取 {l3['blocks_chosen']:,} 个块共 {l3['relations_held']:,} 条关系作证据块隔离。"
        f"\n\n各删除条件的规模差异很大（图 8c）：评价条件化删除只去掉 {pc(l2['annotation_rows']['share'], 2)} 的注释行，却涉及 "
        f"{pc(l2['query_relations']['gene_affected'] / l2['query_relations']['total'])} 的查询关系的基因；仅实验类证据与仅 IEA 则分别删去 "
        f"{pc(au['variants']['experimental']['annotation_rows']['share'], 0)} 与 {pc(au['variants']['iea_only']['annotation_rows']['share'], 0)}"
        f" 的注释行。新旧方法差的变化（图 8d）：{len(retrain)} 个整链重训条件中 {inc0} 个的区间包含 0；去除突变表型与遗传相互作用类证据"
        f"后变化 {nm.get('delta', 0):+.4f}（{nm.get('ci_low', 0):+.4f} 至 {nm.get('ci_high', 0):+.4f}），约为完整数据收益 "
        f"{nm.get('full_gap', 0):+.3f} 的 {abs(nm.get('delta', 0)) / max(nm.get('full_gap', 1e-9), 1e-9):.0%}；仅 IEA 的重打分变化 "
        f"{iea_r.get('delta', 0):+.3f}，反映用完整 GO 训练的模型只喂电子注释时的输入分布改变。三层定向删除与各自数量匹配的随机删除的"
        f"变化方向和大小相近、都在 0 附近，没有证据表明与评价关系共享文献的注释贡献了额外的收益。")
    X = N["x"]
    txt += (
        f"\n\n按疾病大类展开（图 8e，种子 42，{X['dod_ncat']} 个大类，事后描述），整链重训条件的逐类变化在 {X['dod_ret_range'][0]:+.1f} 至 "
        f"{X['dod_ret_range'][1]:+.1f}（×100）之间，{pc(X['dod_ret_small'], 0)} 的格子绝对值不超过 2.0，定向删除与其匹配随机对照在同一大类上"
        f"同号的比例为 {pc(min(X['dod_sign_agree']), 0)}–{pc(max(X['dod_sign_agree']), 0)}（随机时约为 50%），更像是小样本波动而不是"
        f"系统性依赖；只有“仅 IEA”重打分在 {X['dod_iea_neg']} 个大类中为负，方向一致。")
    leg = ("**a** GO 生产记录按证据代码的数量（深色有 PMID，浅色无 PMID），括号内为代码类别，右侧为无 PMID 的比例。**b** 左：PMID—关系"
           "二部图连通块的大小分布（双对数轴，红点为巨块）；右：按大小降序累计的块所覆盖的有 PMID 关系比例。**c** 各删除条件在五种分母"
           "上的删除比例（对数色标）：引文、注释行、唯一基因–术语对、受影响基因、受影响的查询关系；* 为使用了评价关系 PMID 的审计条件。"
           "**d** “新旧方法差”（部署减仅通道 LGB）在删减数据与完整数据之间的变化（种子 42，整链重训；方块为匹配随机对照，灰色为只替换"
           "推理输入的重打分敏感性）。**e** 同一变化按 MeSH 大类展开（×100；测试疾病 ≥60 的大类）：行为 11 个删除条件，列按相似性做平均"
           "连接聚类（上方为聚类树与各类疾病数），右列为全部疾病；蓝色表示删减后新方法的优势扩大，红色表示缩小（事后描述）。"
           f"{LEG_STAT}。")
    return "\n\n".join(["### 3.9 功能证据来源（E4）", txt,
                        recaption(table4(R), "表 4　独立性与外部资格", "表 8　证据删除与外部资格"),
                        fig("Fig8_evidence_sources", "图 8", "GO 证据来源、证据块与删除敏感性", leg)])


def sec_e89(R: dict, N: dict) -> str:
    ext = N["ext"]

    def ec(b, sub):
        r = _row(ext, b=b, subset=sub)
        return f"{r['delta']:+.3f}（{r['ci_low']:+.3f} 至 {r['ci_high']:+.3f}）" if r else "—"

    fl = N["fields"]
    gshared = _row(fl, field="shared_original_evidence", source="GenCC", value="是").get("share")
    cunk = _row(fl, field="shared_original_evidence", source="ClinVar", value="未知").get("share")
    hc = g(R, "historical_contrasts", default={}) or {}
    hist = N["hist"]
    x1n = int(_row(hist, task="x1", method="deployed").get("n", 0))
    x2n = int(_row(hist, task="x2", method="deployed").get("n", 0))
    sup = N["sup"]

    def sv(m, k):
        r = _row(sup, method=m, k=k)
        return f"{r['recall10']:.3f}" if r else "—"

    ex = N["expr"]
    sig = ex[(ex["p_value"] < 0.05) & ex["set"].str.startswith("v8_deployed_s42")]
    sig_ch = ex[(ex["p_value"] < 0.05) & ex["set"].str.startswith("channel")]
    txt = (
        f"外部 618 个疾病（2,441 个关系；V7.1 时冻结，已用于设计）的再分析方向与内部结果一致（图 9a）：部署模型相对仅通道 LightGBM 在合并"
        f"集合上为 {ec('channel_lgb_v6', 'union')}，GenCC 与 ClinVar 子集分别为 {ec('channel_lgb_v6', 'GenCC')} 与 "
        f"{ec('channel_lgb_v6', 'ClinVar')}，文献独立的子集为 {ec('channel_lgb_v6', 'independent_evidence')}；相对 V7.1 直接融合的差值在"
        f"四个子集中都包含 0。三个资格字段（图 9b）显示：没有外部关系出现在训练疾病中；GenCC 关系中 {pc(gshared, 0)} 与训练关系共享原始文献，"
        f"ClinVar 关系 {pc(cunk, 0)} 没有可比对的文献，文献独立性只能在可追溯的范围内判断。"
        f"\n\n2023-12-31 截止的历史任务在截止前资源上重训了全部模型（图 9c）：X2（{x2n} 个已有关系疾病的新关系）上部署模型与仅通道 LightGBM "
        f"的差值为 {ci(hc.get('x2_deployed_vs_channel_lgb_v6'))}，X1（{x1n} 个截止时无关联监督的疾病）上为 "
        f"{ci(hc.get('x1_deployed_vs_channel_lgb_v6'))}；在时间冻结条件下没有看到内部交叉验证中的稳定提升。"
        f"\n\n应用任务（图 9d–e）：533 个疾病的支持任务中，仅用 V8.1 语义（k=0）的 Recall@10 为 {sv('sem_v8', 0)}，高于 V6 语义的 "
        f"{sv('sem_v6', 0)}；显露 5 个支持基因后，V8.1 语义与 RWR 的组合为 {sv('joint_v8_w0.25', 5)}，V6 组合为 {sv('joint_v6', 5)}，"
        f"单独 RWR 与量子游走为 {sv('rwr_v8list', 5)} 与 {sv('qwalk_v8list', 5)}。四个表达队列中，V8.1 的冻结基因集有 {len(sig)} 个"
        f"“队列 × 对比 × Top-K”组合的富集 P < 0.05，仅通道 LightGBM 为 {len(sig_ch)} 个；表达结果只支持相关性。")
    X = N["x"]
    x2, x1 = X["hp"]["x2"], X["hp"]["x1"]
    s1, s5 = X["sup_pairs"].get(1, {}), X["sup_pairs"].get(5, {})
    txt += (
        f"\n\n逐对看历史任务（图 9f，事后描述）：X2 的 {x2['pairs']} 对中部署模型名次更靠前的 {x2['deployed_better']} 对、仅通道 LightGBM 更靠前的 "
        f"{x2['channel_better']} 对，其余 {x2['tie']} 对相同（其中 {x2['both_unlisted']} 对两者都未入选）；进入前十的分别为 {x2['top10_deployed']} "
        f"与 {x2['top10_channel']} 对。X1 的 {x1['pairs']} 对中两者各有 {x1['deployed_better']} 与 {x1['channel_better']} 对更靠前，前十 "
        f"{x1['top10_deployed']} 对 {x1['top10_channel']}。部署模型把略多的新关系带进名单（X2 {x2['listed_deployed']} 对 "
        f"{x2['listed_channel']}，X1 {x1['listed_deployed']} 对 {x1['listed_channel']}），但没有把它们推进前十，这与图 9c 的零差值一致。"
        f"支持任务的逐病比较（图 9g）则稳定偏向 V8.1 语义：显露 1 个支持基因时 {s1.get('up', 0)} 个疾病的 Recall@10 更高、{s1.get('down', 0)} 个更低"
        f"（{s1.get('same', 0)} 个相同），显露 5 个时为 {s5.get('up', 0)}、{s5.get('down', 0)} 与 {s5.get('same', 0)}。")
    leg = ("**a** 外部再分析：部署模型相对三种对照的 Recall@10 差值，按合并、GenCC、ClinVar 与文献独立子集分行（括号内为疾病数）。"
           "**b** 外部关系的三个资格字段按来源的构成。**c** 历史任务（截止 2023-12-31）：X2 与 X1 上各方法的 Recall@10（条形为均值、"
           "竖线为 95% 区间）及部署模型相对仅通道 LGB 的差值。**d** 支持任务（533 个疾病，显露 k 个支持基因；带为 95% 置信带）。"
           "**e** 四个表达队列中冻结基因集的富集（颜色：选中基因 |t| 高于匹配背景的概率，0.5 为无富集；圆面积 ∝ −log10 P；黑圈 P < 0.05）。"
           "**f** 历史任务每个测试关系（X2 圆点、X1 三角；种子 42）在仅通道 LGB（自身 V6 500）与部署模型（严格 500）中的名次，对数轴，"
           "未入选的关系画在分隔带外；对角线下方（蓝底）为部署模型更靠前，上方（橙底）为仅通道更靠前；点位加了很小的对数抖动以免重叠。"
           "**g** 支持任务逐病 Recall@10（显露 k = 1 与 5 个支持基因；k = 1 为 10 次抽样的平均，k = 5 为 1 次），平方根坐标；蓝点为 V8.1 语义 + "
           f"RWR 更高，橙点为 V6 语义 + RWR 更高，灰点相同，灰底为点密度。f–g 为事后描述。{LEG_STAT}。")
    return "\n\n".join(["### 3.10 外部再分析、历史截止与应用任务（E8、E9）", txt,
                        fig("Fig9_external_temporal_applications", "图 9", "外部再分析、历史截止与应用任务", leg)])


def sec_cases(N: dict) -> str:
    rows = N["case_rows"]
    dis = pd.read_parquet(PUB / "diseases.parquet").set_index("disease_id")
    parts = []
    for c in N["cases"]:
        d = c["disease_id"]
        rr = rows[rows["disease_id"] == d]
        dep = rr[(rr["method"] == "deployed") & rr["held_out_positive"]]
        chn = rr[(rr["method"] == "channel_lgb_v6") & rr["held_out_positive"]]
        cold = dep[dep["reference_path_cold"]]["symbol"].tolist()
        name = dis.loc[d, "disease_name"]
        s = (f"{name}（{d}）：部署模型前十中 {len(dep)} 个为留出阳性（{'、'.join(dep['symbol'])}"
             + (f"；其中 {'、'.join(cold)} 为参考路径冷" if cold else "") + "）" if len(dep) else
             f"{name}（{d}）：部署模型前十中没有留出阳性")
        s += (f"，仅通道 LightGBM 为 {len(chn)} 个（{'、'.join(chn['symbol'])}）" if len(chn) else
              "，仅通道 LightGBM 的前十中没有留出阳性")
        parts.append(s)
    txt = ("按作图前写定的规则选取六个案例（图 10）：在至少有 3 个留出阳性的测试疾病中，C16、C10、C18、C04 四个大类各取前十命中增益最大者，"
           "另取全体中损失最大的两个。" + "；".join(parts) + "。四个增益案例中，部署模型前十里的留出阳性多为功能上相近的一组基因"
           "（肌小节蛋白，神经退行相关蛋白，胰岛素与瘦素信号，经典的癌基因、抑癌基因与增殖信号），与“疾病—功能语义”的机制一致。两个损失案例中，"
           "仅通道模型依靠基因 HPO、表型与本体来源（视网膜色素变性）或文本、层级与本体来源（肝肿瘤，该病没有 HPO 描述）找回的阳性被挤出前十：视网膜色素变性中部署模型的前十同样集中在视网膜功能相关基因，"
           "但多数不在该疾病的 CTD 标注阳性中；肝肿瘤中部署模型的前十偏向肝脏代谢相关基因。“未被标注”与“错误”需要区分，这正是"
           "160 对盲审材料要回答的问题；案例只用于说明模型行为，不作为发现的证据。")
    nets = {n["disease_id"]: n for n in N["x"]["nets"]}
    hcm, ad, rp = nets.get("MESH:D024741", {}), nets.get("MESH:D000544", {}), nets.get("MESH:D012174", {})
    txt += (
        f"\n\n把两种方法前十的并集画成基因网络（图 10b）可以看到差别的结构。家族性肥厚型心肌病中，部署模型前十内部有 "
        f"{hcm.get('phys_dep', 0)} 条 STRING 物理互作边，全部在肌小节蛋白之间，仅通道模型的前十内部为 {hcm.get('phys_ch', 0)} 条；"
        f"阿尔茨海默病中两种方法共享 {len(ad.get('shared', []))} 个基因（{'、'.join(ad.get('shared', []))}），部署模型另外加入的 APOE 与 APP "
        f"同共享基因之间有物理互作（APOE—MAPT，APP—PSEN1、APP—PRNP、APP—MAPT），CLU 经 APOE 与 APP 相连（前十内部物理互作："
        f"部署 {ad.get('phys_dep', 0)} 条，仅通道 {ad.get('phys_ch', 0)} 条）；视网膜色素变性中，部署模型"
        f"独有的视蛋白基因彼此以 GO 相似性相连，而仅通道模型找回的 6 个阳性（NRL、CNGA1、RPGR、UNC119、CRB1、RPGRIP1）在 V6 中都有较强的"
        f"基因 HPO 通道信号，多数还有表型或本体信号，彼此之间的 STRING 物理互作很少（前十内部：部署 {rp.get('phys_dep', 0)} 条、仅通道 "
        f"{rp.get('phys_ch', 0)} 条）。网络只用于展示，"
        f"不参与任何评价或选择。")
    leg = ("**a** 每个案例两行：上为部署模型、下为仅通道 LGB 的前 10 名（种子 42，基因符号）；彩色格为被留出的真实关联（CTD marker/mechanism），"
           "黑框与 ° 表示该阳性在评价上下文中为参考路径冷，浅灰为未被标注为关联的基因。选择规则见正文，作图前写定。**b** 三个案例的前 10 "
           "基因网络：节点为两种方法前 10 的并集（左弧：仅部署前 10；右弧：仅通道前 10；中列：两者共有），节点内数字为该方法前 10 中的名次"
           "（共有基因为 部署/仅通道），节点面积 ∝ log(1 + GO 基因–术语对数)，黑圈为留出阳性（° 为参考路径冷）；边为 STRING v12 物理互作"
           "（深灰）与功能关联（浅灰）以及 GO 术语集 Jaccard ≥ 0.2（紫色虚线）。")
    return "\n\n".join(["### 3.11 案例分析", txt,
                        fig("Fig10_case_studies", "图 10", "六个测试疾病的前 10 名、留出关联与基因网络", leg)])


def sec_misc(R: dict, N: dict) -> str:
    sem = {r["method"]: r for r in g(R, "semantic", "table", default=[]) or []}
    comp = g(R, "comparators", default={}) or {}
    kd = comp.get("kdgene", {})
    sg = comp.get("sign", {})
    rep = g(R, "reproduction", default={}) or {}
    acc = json.loads((paths8.AUDIT / "acceptance_ten_checks_v8.json").read_text(encoding="utf-8"))
    arc = json.loads((paths8.AUDIT / "archive_comparison_v8.json").read_text(encoding="utf-8"))

    def sr(k):
        r = sem.get(k, {})
        return f"{r.get('recall10', 0):.3f}" if r else "—"

    txt = (
        f"语义方法的单独排名（全背景，种子 42；图 S4）：教师 $s_F$（分段文本）为 {sr('教师 s_F（分段文本）')}，V7.1 核心 $r$ 为 "
        f"{sr('V7.1 核心 r')}，无学习 SapBERT 功能相似为 {sr('无学习 SapBERT 功能相似')}，线性双塔 {sr('线性双塔')}，"
        f"ProtST 零样本 {sr('ProtST 零样本')}；分段文本与 V7.1 短文本、二级掩蔽文本之间的差值都很小。外部方法在同一严格 500 上重训："
        f"KDGene 的 Recall@10 为 {f(kd.get('recall10'))}，部署模型相对其 {ci(kd.get('deployed_minus'))}；SIGN 式为 "
        f"{f(sg.get('recall10'))}，部署模型相对其 {ci(sg.get('deployed_minus'))}。"
        f"\n\n稳定性与复现：各外层 × 种子的 Recall@10 见图 S3a，方法之间的排序在 15 个组合中基本不变；逐外层看（图 S3b），部署模型相对"
        f"仅通道 LightGBM 更好的疾病占 {pc(N['x']['s3_better'][0], 0)}–{pc(N['x']['s3_better'][1], 0)}、更差的占 "
        f"{pc(N['x']['s3_worse'][0], 0)}–{pc(N['x']['s3_worse'][1], 0)}，{N['x']['s3_ci_pos']} 个外层的差值区间下界大于 0；部署模型逐病 "
        f"Recall@10 在种子之间的 Pearson 相关为 {min(r['pearson_r'] for r in N['x']['s3c']):.2f}–{max(r['pearson_r'] for r in N['x']['s3c']):.2f}，"
        f"{pc(min(r['identical_share'] for r in N['x']['s3c']), 0)}–{pc(max(r['identical_share'] for r in N['x']['s3c']), 0)} 的疾病完全相同（图 S3c）。教师的疾病向量在 t-SNE 中按"
        f"疾病大类形成局部聚集（图 S5a；{N['tsne']['n']:,} 个疾病）；已知关联的疾病—基因对在语义空间中的余弦明显高于随机有 GO 的基因"
        f"（AUC = {N['tsne_auc']:.3f}，{N['tsne']['pairs_cosine']:,} 对，图 S5b），这些关联在训练中可见，只说明语义空间的结构，"
        f"不作为测试性能证据。十项整链验收{'全部通过' if acc.get('all_pass') else '见审计'}；V5/V6/V7.1 档案首尾哈希"
        f"{'一致' if arc.get('archives_untouched') else '见审计'}。独立虚拟环境 `.venv_repro` 在计算前被 Windows 应用程序控制策略阻止"
        f"加载 SciPy 的扩展模块，按安全规则未修改系统设置；第 0 折种子 42 子链改在主环境以全新缓存从冻结输入重算，候选名单一致比例 "
        f"{pc(rep.get('pools_identical_share'))}、前十一致比例 {pc(rep.get('top10_identical_share'))}、部署分数最大差 "
        f"{rep.get('deployed_scores_maxabs', 0):.0e}——这检验的是流程的确定性而不是环境的独立性。160 对双人盲审材料已生成，审核本身需要"
        f"两名独立审阅者，本轮未进行。计算资源与运行时间线见图 S6。")
    return "\n\n".join(["### 3.12 语义方法比较、稳定性、复现与资源", txt,
                        recaption(__import__("gpu8v8final.manuscript8", fromlist=["x"]).table5(R), "表 5　运行、验收与复现状态",
                                  "表 9　运行、验收与复现状态")])


def results(R: dict, N: dict) -> str:
    return "\n\n".join(["## 3 结果", sec_landscape(N), sec_method(R, N), sec_dev(R, N), sec_e1(R, N), sec_budget(R, N),
                        sec_strata(R, N), sec_e2(R, N), sec_e56(R, N), sec_e4(R, N), sec_e89(R, N), sec_cases(N),
                        sec_misc(R, N)])

# ----------------------------------------------------------------------------- discussion / supplement / writer
def discussion(R: dict, N: dict) -> str:
    b = N["s5b"]
    rich = _row(b, stratum="≥100", method="deployed").get("hit10")
    zero = _row(b, stratum="0", method="deployed").get("hit10")
    fam = N["fi_fam"]
    extra = (
        f"**收益沿着注释覆盖分布。** 事后分层给出了“知识覆盖不均衡”在结果中的形状：部署模型把参考路径冷阳性的前十恢复率提高到 "
        f"{pc(_row(N['s5a'], stratum='冷', method='deployed').get('hit10'))}（仅通道模型几乎为零），但绝对水平仍很低；增益最大的是 GO 注释丰富的基因（≥100 个基因–术语对时前十恢复 {pc(rich)}），没有 GO 的基因几乎得不到"
        f"语义收益（{pc(zero)}），有 HPO 与较长文本的疾病增益也明显大于只有名称的疾病。语义通道把“参考关系”的不均衡换成了对“功能注释”"
        f"的依赖；而图 1 显示，这两种知识本身就是同向集中的。这与序列补位、冷正例加权都没有改变结果的现象一致：两者都试图在"
        f"注释或监督缺失处补上信号，而补位的质量不足以与观察到的 GO 竞争。更有希望的方向是提高缺失处功能预测的质量，或者只在补位"
        f"置信度高时让其进入融合。"
        f"\n\n**二级模型主要在候选融合的顺序上重排。** 部署的候选 A 把约 {fam.get('候选融合', 0):.0%} 的增益放在缺失感知融合名次这一个"
        f"特征上，语义特征合计约 {fam.get('语义', 0):.0%}，V6 通道合计约 {fam.get('V6 检索通道', 0):.0%}。这与门控纠偏器的诊断一致：纠偏器的"
        f"验证曲线在几十轮内达到峰值后回落，说明从每病 64 个抽样的未标注基因中学到的“正例高于未标注”信号很快被噪声覆盖；在候选名单"
        f"已经由缺失感知融合排好序时，直接在完整名单上优化前十的 lambdarank 更稳健。"
        f"\n\n**案例与局限的对应。** 两个损失案例中，仅通道模型已经找到的专门基因被语义候选挤出前十，提示在检索通道本身已经很强的"
        f"疾病上，语义候选带来的可能是替换而不是补充；历史任务 X2（已有关系疾病的新关系）上没有收益，方向与此一致。")
    return "## 4 讨论\n\n" + R.get("_txt_discussion", "") + "\n\n" + extra


def limits(R: dict) -> str:
    extra = ("第七，图 2e、图 3c 与 3g–i、图 4a 与 4d–g、图 5、图 6d–h、图 7f–h、图 8e、图 9f–g、图 10 与图 S3b–c、图 S5 的分析在登记之后"
             "进行，只使用冻结的测试预测（图 6e–h 另用保存的模型重建特征矩阵），结果为描述性，不作显著性判定；分层数目较多，个别层的区间"
             "应按多次比较的背景理解。TreeSHAP 只解释一个外层上下文与一个种子的模型，归因是对模型的解释而不是因果效应。案例按作图前写定的"
             "规则选取，网络只用于展示，均不作为发现的证据。")
    return "## 5 局限\n\n" + R.get("_txt_limits", "") + "\n\n" + extra


def conclusion(R: dict) -> str:
    return ("## 6 结论\n\n" + R.get("_txt_conclusion", "") +
            "事后分析进一步表明，语义检索的收益沿功能注释的覆盖分布：它缓解了参考路径上的不均衡，但对缺少注释的基因帮助有限。")


def availability() -> str:
    return ("## 数据与代码可用性\n\n代码位于 `gpu8/src/gpu8v8final/`（作图：`pubdata8.py`、`pubdata8b.py`、`pubviz8.py`、`pubfigs8a*.py`、`pubfigs8b.py`、"
            "`pubfigs8c.py`、`pubfigs8x.py`、`pubfigs8s.py`；本稿：`manuscript8v2.py`）；V5、V6、V7.1 的代码与结果档案未改动。逐运行权威结果在 `gpu8/runs_v8/*/result.json`，"
            "汇总账本 `gpu8/manifests_v8/run_ledger_v8.csv`，登记文件 `gpu8/manifests_v8/protocol_v8_registered.json`，数字溯源 "
            "`gpu8/results_v8/numbers_v8_results.json`；作图的中间数据在 `gpu8/results_v8/pub/`，每个图面板的作图数据以 CSV 存于 "
            "`manuscript_v8/figures_v2/source_data/`，图件的矢量版本（PDF/SVG）与 600 dpi PNG 同目录提供。")


def supplement(R: dict, N: dict) -> str:
    import re

    from .manuscript8 import supplement as supp_v1
    base = supp_v1(R).replace("## 补充材料\n\n", "", 1)
    base = base.replace("| background |", "| 全背景 |").replace("| reference |", "| 参考关系基因 |")
    base = re.sub(r"A_nl(\d+)_mc(\d+)", r"\1 叶/最小样本 \2", base)
    tags = {"e5_none": "E5 无补位", "e5_mlp": "E5 两层学生", "e5_linear": "E5 线性学生", "e5_nn": "E5 序列近邻转移",
            "e5_teacher_sS": "E5 教师序列投影", "e5_mlp_shuffled": "E5 两层学生（配对打乱）", "e5_protst": "E5 ProtST 输入学生",
            "e6_cold_gomask": "E6 监督冷 + 删 GO", "e6_cold": "E6 监督冷", "natural_student_on": "自然输入中开启学生",
            "plain_rrf": "普通 RRF", "expr_off": "关闭表达", "expr_random": "随机组织表达"}
    for k in sorted(tags, key=len, reverse=True):
        base = base.replace(f"| {k} |", f"| {tags[k]} |")
    sets = {"lesional_vs_uninvolved_paired": "病灶/非病灶（配对）", "lesional_vs_control": "病灶/对照"}
    for k, v in sets.items():
        base = base.replace(f"| {k} |", f"| {v} |")
    base = re.sub(r"\| v8_deployed_s42_top(\d+) \|", r"| V8.1 部署 Top-\1 |", base)
    base = re.sub(r"\| channel_lgb_v6_s42_top(\d+) \|", r"| 仅通道 LGB Top-\1 |", base)
    base = re.sub(r"\| matched_random_top(\d+) \|", r"| 匹配随机 Top-\1 |", base)
    meth = {"joint_v8_w0.25": "V8.1 语义 + RWR（ω=0.25）", "joint_v6": "V6 语义 + RWR", "qwalk_v6list": "量子游走（V6 名单）",
            "qwalk_v8list": "量子游走（V8.1 名单）", "rwr_v6list": "RWR（V6 名单）", "rwr_v8list": "RWR（V8.1 名单）",
            "sem_v6": "V6 语义单独", "sem_v8": "V8.1 语义单独"}
    for k in sorted(meth, key=len, reverse=True):
        base = base.replace(f"| {k} |", f"| {meth[k]} |")
    hist = {"gene background (19,861) and HGNC-based gene index": "基因背景（19,861）与基于 HGNC 的基因索引",
            "entity index": "实体索引", "protein sequences (MANE 2025)": "蛋白序列（MANE 2025）",
            "static gene attribute": "静态基因属性", "STRING v12.0 physical network": "STRING v12.0 物理互作网络",
            "PPI (not later than the cut-off)": "PPI，不晚于截止", "disease equivalence groups and families": "疾病等价组与家族",
            "split units only": "仅用于划分", "V8.1 hyper-parameters and development choices": "V8.1 超参数与开发选择",
            "frozen settings": "冻结设置", "SapBERT / ESM-2 checkpoints": "SapBERT / ESM-2 检查点",
            "frozen encoders": "冻结编码器"}
    for k, v in hist.items():
        base = base.replace(k, v)
    d = N["s5d"]
    rows = [[f"{r['name']}（{r['code']}）", i0(r["n"]), f(r["deployed"]), f(r["v71_direct_strict"]), f(r["channel_lgb_v6"]),
             f(r["standalone_sF"]), f"{r['delta']:+.3f}（{r['lo']:+.3f} 至 {r['hi']:+.3f}）",
             f"{r['delta_v71']:+.3f}（{r['lo_v71']:+.3f} 至 {r['hi_v71']:+.3f}）"] for _, r in d.iterrows()]
    s7 = ("表 S7　按 MeSH 疾病大类的 Recall@10（测试疾病 ≥60 的大类；三种子平均；事后描述）\n\n" +
          table_md(["大类", "疾病数", "V8.1 部署", "V7.1 直接融合", "仅通道 LGB", "教师 $s_F$ 单独", "部署 − 仅通道", "部署 − V7.1"], rows))
    cr = N["case_rows"]
    rows8 = []
    for c in N["cases"]:
        for m, lab in (("deployed", "V8.1 部署"), ("channel_lgb_v6", "仅通道 LGB")):
            s_ = cr[(cr["disease_id"] == c["disease_id"]) & (cr["method"] == m)].sort_values("rank")
            rows8.append([c["disease_id"], lab, "、".join((f"**{x}**" if h else x) + ("°" if cold else "")
                                                          for x, h, cold in zip(s_["symbol"], s_["held_out_positive"],
                                                                                s_["reference_path_cold"]))])
    s8 = ("表 S8　案例疾病的前 10 名（粗体为留出的真实关联，° 为参考路径冷）\n\n" + table_md(["疾病", "方法", "前 10 名"], rows8))
    sfigs = [
        ("FigS1_teacher_training", "图 S1", "语义教师的训练与对照集选择",
         "**a** 五个外层上下文中教师在同一上下文 $F_{j+1}$ 上的 $s_F$ 全背景 Recall@10 随轮次的变化（颜色深浅为种子；灰线为参考关系基因对照，"
         "种子 42），圆点为最佳轮次。**b** 左：开发集上两种未标注对照集的 $s_F$ Recall@10；右：各外层教师的最佳轮次（实心）与早停轮次（空心）。"),
        ("FigS2_development_selection", "图 S2", "开发集上的配置选择",
         "**a** 各外层的开发阳性覆盖@500 随 RRF 权重 w 的变化（缺失感知与普通 RRF；橙线为 V6 500；黑圈为选定的 $w_j$）。**b** 候选 A 的六个"
         "配置（叶数 × 最小样本）的开发 Recall@10，黑框为部署的配置。**c** 每个外层上基础排序器、最佳纠偏器与最佳候选 A 的开发 Recall@10 与差距。"),
        ("FigS3_fold_seed_stability", "图 S3", "各外层 × 种子的稳定性",
         "**a** 各外层 × 种子的端到端 Recall@10：行为方法，列为外层 · 种子，最右为行均值（统一色标）。**b** 逐外层的逐病比较（三种子"
         "平均）：左为部署模型相对仅通道 LGB 与 V7.1 直接融合的胜 / 平 / 负比例，右为差值与家族 bootstrap 95% 区间（实心圆：对仅通道 LGB；"
         "空心方块：对 V7.1 直接融合）。**c** 部署模型逐病 Recall@10 在种子之间的一致性（六边形分箱，对数色标；r 为 Pearson 相关）。"),
        ("FigS4_semantic_variants", "图 S4", "全背景单独语义排名与文本变体",
         "**a** 各语义方法的 Recall@10（条）与 $R^c$@10（圈），紫色为教师 $s_F$ 的三种文本版本。**b** 相对教师 $s_F$（分段文本）的 Recall@10 "
         f"差值与 95% 区间（与左图同行）。"),
        ("FigS5_embedding", "图 S5", "语义空间的结构",
         "**a** 教师疾病向量 $z_d$ 的 t-SNE（外层 0，种子 42；困惑度 40，GPU 实现），每个小图突出一个 MeSH 大类（括号内为疾病数）。"
         "**b** 已知关联的疾病—基因对与随机有 GO 基因的余弦分布；这些关联在训练中可见，只说明空间结构。"),
        ("FigS6_compute_resources", "图 S6", "计算资源",
         "**a** 账本中各组件运行的完成时间线（点大小 ∝ 计时秒数；中间空档为会话中断）。**b** 各组件的运行数与计时机时。"),
        ("FigS7_evidence_blocks", "图 S7", "证据块关系的恢复与删减数据中的新旧方法差",
         "**a** 被隔离证据块中 2,497 条关系的前十恢复率：空心为完整数据，箭头指向删减数据（蓝：部署模型；橙：仅通道 LGB，只在证据块"
         "条件中重训）。**b** 各删除条件下部署模型相对仅通道 LGB 的 Recall@10 差值；灰带与竖线为完整数据的估计与 95% 区间。"),
    ]
    figs = "\n\n".join(fig(s, n, t, l) for s, n, t, l in sfigs)
    return base + "\n\n" + s7 + "\n\n" + s8 + "\n\n### 补充图\n\n" + figs


def compose(R: dict, N: dict) -> tuple[str, str]:
    from .manuscript8 import title as title_v1
    T = title_v1(R)
    parts = [f"# {T}", "## 摘要", abstract(R, N),
             "**关键词** 疾病基因检索；知识覆盖不均衡；冷启动；功能语义；候选预算；缺失感知融合；袋外评价",
             INTRO, METHODS, experiments(R, N), results(R, N), discussion(R, N), limits(R), conclusion(R), availability(),
             "## 补充材料", supplement(R, N)]
    return T, "\n\n".join(p.strip("\n") for p in parts) + "\n"


def write() -> dict:
    R = load_R()
    N = numbers(R)
    T, md = compose(R, N)
    OUT.mkdir(parents=True, exist_ok=True)
    md_path = OUT / f"{T.split('：')[0]}_V81_修订_{DATE}.md"
    md_path.write_text(md, encoding="utf-8")
    from PIL import Image

    from gpu8v6.build_docx6 import build as build_docx
    small = FIGDIR / "docx_300dpi"          # 300-dpi copies keep the Word file light; 600-dpi/vector stay alongside
    small.mkdir(exist_ok=True)
    for p in FIGDIR.glob("*.png"):
        im = Image.open(p)
        im.resize((im.width // 2, im.height // 2), Image.LANCZOS).save(small / p.name, dpi=(300, 300), optimize=True)
    docx_path = md_path.with_suffix(".docx")
    st = build_docx(md_path, docx_path, small)
    from docx import Document
    from docx.shared import Inches
    doc = Document(str(docx_path))
    for shp in doc.inline_shapes:
        w, h = shp.width, shp.height
        shp.width = Inches(6.3)
        shp.height = int(h * Inches(6.3) / w)
    doc.save(str(docx_path))
    st["picture_width_in"] = 6.3
    dump_json(paths8.REPORTS / "docx_build_report_v8_revision.json", st)
    info = {"generated_utc": now_utc(), "markdown": str(md_path), "docx": str(docx_path),
            "omml": st.get("verified_omml_objects"), "tables": st.get("verified_tables"), "images": st.get("images"),
            "equation_failures": len(st.get("equation_failures", [])), "missing_figures": st.get("missing_figures")}
    dump_json(paths8.REPORTS / "manuscript_revision_v8.json", info)
    return info


if __name__ == "__main__":
    paths8.ensure()
    print(json.dumps(write(), ensure_ascii=False, indent=1))
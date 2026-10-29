"""Figures 9-10 (external / historical / applications; case studies) and supplementary figures S1-S7."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch

from . import paths8
from .pubcolor8 import AXIS, BLUE, CAT, ENTITY, INK, INK2, MUTED, SURFACE, div_cmap_rev, ink_on, seq_cmap
from .pubstyle8 import (FS, FS_S, axmm, fam_ci, fig_mm, heat_table, label_mm, legend_handles, mt, pct, save, style)

PUB = paths8.RESULTS / "pub"
EXT = paths8.RESULTS / "external"
EVAL = paths8.RESULTS / "eval"


def _nums() -> dict:
    return json.loads((paths8.RESULTS / "numbers_v8_results.json").read_text(encoding="utf-8"))


def _fam() -> dict:
    d = pd.read_parquet(PUB / "diseases.parquet")
    return dict(zip(d["disease_id"], d["family_group"]))


# ====================================================================== Figure 9
def fig9() -> dict:
    R = _nums()
    fam = _fam()
    H9 = 220
    fig = fig_mm(183, H9)
    src = {}
    # ---------------------------------------------------- a: external 618 re-analysis
    label_mm(fig, 2, 6, "a", "外部 618 个疾病的再分析（V7.1 冻结集合，已用于设计）")
    con = pd.DataFrame(R["external"]["contrasts"])
    vs_lab = {"channel_lgb_v6": "对仅通道 LGB", "v71_direct_strict": "对 V7.1 直接融合", "B": "对基础排序器 B"}
    sub_lab = {"union": "合并", "GenCC": "GenCC", "ClinVar": "ClinVar", "independent_evidence": "独立证据子集"}
    ax = axmm(fig, 44, 13, 40, 50)
    rows_a, ypos, ylab = [], [], []
    y = 0.0
    for b in ("channel_lgb_v6", "v71_direct_strict", "B"):
        s_ = con[con["b"] == b]
        for sub in ("union", "GenCC", "ClinVar", "independent_evidence"):
            r = s_[s_["subset"] == sub]
            if r.empty:
                continue
            r = r.iloc[0]
            colr = ENTITY.get(b, MUTED) if b != "B" else CAT[3]
            ax.plot([r["ci_low"], r["ci_high"]], [y, y], color=colr, lw=1.0)
            ax.plot([r["delta"]], [y], "o" if sub == "union" else "s", ms=3.8 if sub == "union" else 3.0, color=colr,
                    mec=SURFACE, mew=0.5, zorder=3)
            ypos.append(y)
            ylab.append(f"{vs_lab[b]} · {sub_lab[sub]}（{int(r['n'])}）")
            rows_a.append(r.to_dict())
            y += 1
        y += 0.6
    ax.axvline(0, color=AXIS, lw=0.7)
    ax.set_yticks(ypos, ylab, fontsize=FS_S - 1.2)
    ax.set_ylim(y - 0.4, -0.8)
    ax.set_xlabel("部署模型减对照的 Recall@10 差值", fontsize=FS_S - 0.5)
    style(ax, grid="x", left=False)
    ax.tick_params(axis="y", length=0)
    src["a"] = pd.DataFrame(rows_a)

    # ---------------------------------------------------- b: eligibility fields of the external pairs
    label_mm(fig, 92, 6, "b", "外部关系的三个资格字段（2,441 对）")
    pf = pd.read_parquet(EXT / "external618" / "external_pairs_fields.parquet")
    ax = axmm(fig, 116, 13, 64, 40)
    specs = [("seen_in_training_disease", "训练中见过该疾病", {True: "是", False: "否"}),
             ("same_relation_seen", "邻近疾病见过同一关系", {True: "是", False: "否"}),
             ("shared_original_evidence", "与训练共享原始文献", {"yes": "是", "no": "否", "unknown": "未知"})]
    colmap = {"是": CAT[7], "否": BLUE[450], "未知": "#d6d5cf"}
    rows_b = []
    yy = 0
    for col, lab, mp in specs:
        for src_name in ("GenCC", "ClinVar"):
            s_ = pf[pf["source"] == src_name][col].map(mp)
            left = 0.0
            for k in ("否", "是", "未知"):
                v = (s_ == k).mean()
                if v <= 0:
                    continue
                ax.barh(yy, v, left=left, height=0.64, color=colmap[k], edgecolor=SURFACE, linewidth=0.6, zorder=2)
                if v > 0.07:
                    ax.text(left + v / 2, yy, f"{v:.0%}", ha="center", va="center", fontsize=FS_S - 1.4,
                            color=ink_on(colmap[k]))
                left += v
                rows_b.append({"field": col, "source": src_name, "value": k, "share": v, "n": int(len(s_))})
            ax.text(-0.02, yy, f"{src_name}（{len(s_):,}）", ha="right", va="center", fontsize=FS_S - 1.3, color=INK2,
                    transform=ax.get_yaxis_transform())
            yy += 1
        ax.text(0.0, yy - 2.62, lab, ha="left", va="bottom", fontsize=FS_S - 0.9, color=INK, fontweight="bold",
                transform=ax.get_yaxis_transform())
        yy += 0.85
    ax.set_yticks([])
    ax.set_ylim(yy - 0.6, -1.1)
    ax.set_xlim(0, 1)
    pct(ax, "x")
    ax.set_xlabel("关系比例", fontsize=FS_S - 0.5)
    ax.legend(handles=legend_handles([("否", BLUE[450], "box"), ("是", CAT[7], "box"), ("未知（无可比对文献）", "#d6d5cf", "box")]),
              loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=FS_S - 1.2)
    style(ax, grid=None, left=False)
    src["b"] = pd.DataFrame(rows_b)

    # ---------------------------------------------------- c: historical cut-off
    label_mm(fig, 2, 78, "c", "2023-12-31 截止的历史任务（全部模型重训）")
    hp = pd.read_parquet(EXT / "historical" / "s42" / "historical_per_disease.parquet")
    # bar order keeps orange (channel-only) and yellow (B) apart: their normal-vision dE is below the floor
    meths = [("deployed", "V8.1 部署"), ("channel_lgb_v6", "仅通道 LGB"), ("standalone_sF", "教师 $s_F$ 单独"),
             ("B", "基础排序器 B")]
    rows_c = []
    for k, (task, tl) in enumerate((("x2", "X2：已有关系疾病\n的新关系"), ("x1", "X1：截止时无关联\n监督的疾病"))):
        ax = axmm(fig, 13 + k * 30, 90, 22, 42)
        t = hp[hp["task"] == task]
        for i, (m, lab) in enumerate(meths):
            s_ = t[t["method"] == m].set_index("disease_id")["recall10"]
            e = fam_ci(s_.to_numpy(), np.array([fam.get(x, x) for x in s_.index]), 2000)
            colr = ENTITY.get(m, CAT[3] if m == "B" else MUTED)
            ax.bar(i, e[0], width=0.62, color=colr, zorder=2)
            ax.plot([i, i], [e[1], e[2]], color=INK2, lw=0.7, zorder=3)
            rows_c.append({"task": task, "method": m, "recall10": e[0], "lo": e[1], "hi": e[2], "n": len(s_)})
        n_d = int(t[t["method"] == "deployed"]["disease_id"].nunique())
        ax.set_xticks(range(len(meths)), [mt(m[1]) for m in meths], rotation=45, ha="right", fontsize=FS_S - 1.3)
        ax.set_title(f"{tl}（{n_d} 病）", fontsize=FS_S - 1.1, pad=3, fontweight="normal", loc="center")
        if k == 0:
            ax.set_ylabel("Recall@10", fontsize=FS_S - 0.3)
        ax.set_ylim(0, 0.42 if task == "x1" else 0.14)
        style(ax, grid="y")
        hc = R["historical_contrasts"][f"{task}_deployed_vs_channel_lgb_v6"]
        ax.text(0.5, 0.97, f"部署 − 仅通道：\n{hc['delta']:+.3f} [{hc['ci_low']:+.3f}, {hc['ci_high']:+.3f}]",
                transform=ax.transAxes, ha="center", va="top", fontsize=FS_S - 1.6, color=INK2, linespacing=1.2)
    src["c"] = pd.DataFrame(rows_c)

    # ---------------------------------------------------- d: support tasks
    label_mm(fig, 70, 78, "d", "支持任务（533 病，显露 k 个支持基因）")
    ep = pd.read_parquet(EXT / "support" / "support533_test_episodes.parquet")
    ax = axmm(fig, 81, 86, 42, 40)
    ser = [("joint_v8_w0.25", "V8.1 语义 + RWR", CAT[0], "o"), ("joint_v6", "V6 语义 + RWR", CAT[1], "s"),
           ("rwr_v8list", "RWR（V8.1 名单）", "#6f6e69", "^"), ("qwalk_v8list", "量子游走（V8.1 名单）", "#a8a7a1", "v")]
    rows_d = []
    for m, lab, colr, mk in ser:
        s_ = ep[ep["method"] == m]
        xs, ys, los, his = [], [], [], []
        for kk, g in s_.groupby("k"):
            per = g.groupby("disease_id")["recall10"].mean()
            e = fam_ci(per.to_numpy(), np.array([fam.get(x, x) for x in per.index]), 1000)
            xs.append(kk)
            ys.append(e[0])
            los.append(e[1])
            his.append(e[2])
            rows_d.append({"method": m, "k": kk, "recall10": e[0], "lo": e[1], "hi": e[2], "n": len(per)})
        ax.fill_between(xs, los, his, color=colr, alpha=0.12, lw=0)
        ax.plot(xs, ys, color=colr, lw=1.1, marker=mk, ms=3.4, mec=SURFACE, mew=0.5, label=lab)
    for m, lab, colr in (("sem_v8", "V8.1 语义单独（k = 0）", CAT[6]), ("sem_v6", "V6 语义单独（k = 0）", "#c9c8c2")):
        s_ = ep[(ep["method"] == m) & (ep["k"] == 0)]
        per = s_.groupby("disease_id")["recall10"].mean()
        e = fam_ci(per.to_numpy(), np.array([fam.get(x, x) for x in per.index]), 1000)
        ax.plot([0], [e[0]], "D", ms=3.8, color=colr, mec=SURFACE, mew=0.5, label=lab, zorder=4)
        ax.plot([0, 0], [e[1], e[2]], color=colr, lw=0.9)
        rows_d.append({"method": m, "k": 0, "recall10": e[0], "lo": e[1], "hi": e[2], "n": len(per)})
    ax.set_xticks([0, 1, 3, 5])
    ax.set_xlabel("显露的支持基因数 k", fontsize=FS_S - 0.4)
    ax.set_ylabel("Recall@10", fontsize=FS_S - 0.3)
    fig.legend(*ax.get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(100 / 183, 1 - 139.5 / H9),
               ncol=2, fontsize=FS_S - 1.4, handlelength=1.4, columnspacing=0.8)
    style(ax, grid="y")
    src["d"] = pd.DataFrame(rows_d)

    # ---------------------------------------------------- e: expression cohorts (bubble grid)
    label_mm(fig, 128, 78, "e", "四个表达队列的富集（冻结基因集）")
    ex = pd.read_csv(EXT / "expression_v8_table.csv")
    ex["row"] = ex["accession"] + "\n" + ex["contrast"].map({"lesional_vs_uninvolved_paired": "病灶/非病灶配对",
                                                              "lesional_vs_control": "病灶/对照"}).fillna(ex["contrast"])
    order_sets = [("v8_deployed_s42_top20", "V8.1 T20"), ("v8_deployed_s42_top50", "V8.1 T50"),
                  ("channel_lgb_v6_s42_top20", "仅通道 T20"), ("channel_lgb_v6_s42_top50", "仅通道 T50"),
                  ("matched_random_top20", "随机 T20"), ("matched_random_top50", "随机 T50")]
    rows_e = ex["row"].drop_duplicates().tolist()
    ax = axmm(fig, 150, 88, 31, 38)
    from matplotlib.colors import Normalize, to_hex
    nrm = Normalize(0.35, 0.65)
    cm = div_cmap_rev()
    for i, rw in enumerate(rows_e):
        for jx, (sname, _) in enumerate(order_sets):
            r = ex[(ex["row"] == rw) & (ex["set"] == sname)]
            if r.empty:
                continue
            r = r.iloc[0]
            size = 6 + 16 * min(-np.log10(r["p_value"]), 3.5)
            ax.scatter(jx, i, s=size, color=to_hex(cm(nrm(r["enrichment"]))), edgecolors=INK if r["p_value"] < 0.05
                       else SURFACE, linewidths=0.8 if r["p_value"] < 0.05 else 0.4, zorder=3)
    ax.set_xticks(range(len(order_sets)), [s[1] for s in order_sets], rotation=90, fontsize=FS_S - 1.5)
    ax.set_yticks(range(len(rows_e)), rows_e, fontsize=FS_S - 1.8, linespacing=1.0)
    ax.set_xlim(-0.6, len(order_sets) - 0.4)
    ax.set_ylim(len(rows_e) - 0.5, -0.5)
    ax.tick_params(length=0)
    for s_ in ax.spines.values():
        s_.set_visible(False)
    ax.grid(True, color="#efeee9", lw=0.4)
    ax.set_axisbelow(True)
    cax = axmm(fig, 150, 140.5, 31, 1.5)
    import matplotlib as mpl
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=nrm, cmap=cm), cax=cax, orientation="horizontal")
    cb.set_label("富集（|t| 高于匹配背景的概率；0.5 = 无富集）", fontsize=FS_S - 1.9, labelpad=1)
    cb.ax.tick_params(labelsize=FS_S - 1.7, length=1.2)
    cb.outline.set_visible(False)
    fig.text(150 / 183, 1 - 150.5 / H9, "圆面积 ∝ −log10 P；黑圈 P < 0.05；只支持相关性", fontsize=FS_S - 1.8, color=INK2)
    src["e"] = ex[ex["set"].isin([s[0] for s in order_sets])]

    # ---------------------------------------------------- f/g: pair- and disease-level views (post hoc, descriptive)
    from .pubfigs8x import p_hist_pairs, p_support_scatter
    label_mm(fig, 2, 160, "f", "历史任务的逐对名次（X2 与 X1，种子 42）")
    src["f"] = p_hist_pairs(fig, 16, 167, 44, 36)
    label_mm(fig, 76, 160, "g", "支持任务的逐病比较（533 病，抽样平均）")
    src["g"] = pd.concat([p_support_scatter(fig, 88, 168, 38, 38, 1),
                          p_support_scatter(fig, 139, 168, 38, 38, 5, ylabel=False)])
    fig.legend(handles=legend_handles([("V8.1 语义 + RWR 更高", CAT[0], "dot"), ("V6 语义 + RWR 更高", CAT[1], "dot"),
                                       ("相同", "#a8a7a1", "dot"), ("点密度", "#d6d5cf", "box")]),
               loc="upper center", bbox_to_anchor=(133 / 183, 1 - 213.2 / H9), ncol=4, fontsize=FS_S - 1.6,
               handlelength=1.0, columnspacing=0.9)
    fig.text(133 / 183, 1 - 218.2 / H9, "逐病 Recall@10 为 10 次抽样（k = 5 为 1 次）的平均；坐标为平方根刻度", ha="center",
             fontsize=FS_S - 1.9, color=INK2)
    return save(fig, "Fig9_external_temporal_applications", src)


# ====================================================================== Figure 10 (case studies)
def _top10(pred: dict, i: int) -> list[int]:
    v = pred["valid"][i] if "valid" in pred else np.isfinite(pred["score"][i])
    g = pred["genes"][i][v].astype(np.int64)
    s = pred["score"][i][v].astype(np.float64)
    return g[np.lexsort((g, -s))][:10].tolist()


def fig10() -> dict:
    from .evaluate8 import load_pred
    from .pubdata8 import MESH_CAT
    qr = pd.read_parquet(PUB / "query_ranks.parquet")
    q42 = qr[qr["seed"] == 42]
    qc = pd.read_parquet(PUB / "query_context.parquet")
    genes = pd.read_parquet(PUB / "genes.parquet").set_index("gene_row")
    dis = pd.read_parquet(PUB / "diseases.parquet").set_index("disease_id")
    piv = q42[q42["method"].isin(["deployed", "channel_lgb_v6"])].assign(
        hit=lambda d: ((d["rank"] > 0) & (d["rank"] <= 10)).astype(int))
    h = piv.groupby(["disease_id", "method"])["hit"].sum().unstack()
    n = q42[q42["method"] == "deployed"].groupby("disease_id").size()
    h = h.assign(n=n, fold=q42.groupby("disease_id")["fold"].first())
    h["delta"] = h["deployed"] - h["channel_lgb_v6"]
    h = h[h["n"] >= 3].copy()
    h["cats"] = dis.reindex(h.index)["cats"].fillna("")
    # selection rule (declared in the legend): largest gain per category in C16, C10, C18, C04; two largest losses
    chosen = []
    for code in ("C16", "C10", "C18", "C04"):
        c_ = h[h["cats"].str.contains(code) & ~h.index.isin([c[0] for c in chosen])]
        c_ = c_.sort_values(["delta", "n"], ascending=[False, False])
        if len(c_):
            chosen.append((c_.index[0], f"{MESH_CAT[code]}：增益最大", code))
    loss = h[~h.index.isin([c[0] for c in chosen])].sort_values(["delta", "n"], ascending=[True, False])
    for d in loss.index[:2]:
        chosen.append((d, "全体：损失最大", ""))
    cold = qc.set_index(["disease_id", "gene_row"])["cold"]
    H_ = 226
    fig = fig_mm(183, H_)
    label_mm(fig, 2, 6, "a", "案例：测试疾病的前 10 名与被留出的真实关联（种子 42）")
    rows = []
    cw, rh = 13.2, 4.3
    x0 = 26.0
    ytop = 12.5
    for k in range(10):
        xx = 2 + x0 - 2 + k * (cw + 1.0) + cw / 2
        fig.text(xx / 183, 1 - 11.0 / H_, f"第 {k + 1} 名", ha="center", va="baseline", fontsize=FS_S - 1.3, color=MUTED)
    for ci, (d, why, code) in enumerate(chosen):
        j = int(h.loc[d, "fold"])
        dd = paths8.FORMAL / "main" / "s42" / f"fold{j}"
        pos_rows = set(q42[(q42["disease_id"] == d) & (q42["method"] == "deployed")]["gene_row"].astype(int))
        yb = ytop + ci * 21.0
        name = str(dis.loc[d, "disease_name"])
        name = name if len(name) <= 48 else name[:46] + "…"
        fig.text(2 / 183, 1 - (yb + 2.8) / H_, f"{name}", fontsize=FS_S - 0.3, fontweight="bold", color=INK,
                 va="baseline")
        fig.text(2 / 183, 1 - (yb + 6.1) / H_,
                 f"{d} · {why} · 留出阳性 {int(h.loc[d, 'n'])} 个 · 前 10 命中：部署 {int(h.loc[d, 'deployed'])} / "
                 f"仅通道 {int(h.loc[d, 'channel_lgb_v6'])}", fontsize=FS_S - 1.4, color=INK2, va="baseline")
        ax = axmm(fig, 2, yb + 7.4, 179, 2 * rh + 1.2)
        ax.set_xlim(0, 179)
        ax.set_ylim(2 * rh + 1.2, 0)
        ax.axis("off")
        for ri, (m, lab, colr) in enumerate((("deployed", "V8.1 部署", CAT[0]), ("channel_lgb_v6", "仅通道 LGB", CAT[1]))):
            pred = load_pred(dd / f"pred_{m}__test.npz")
            ids = [str(x) for x in pred["ids"]]
            top = _top10(pred, ids.index(d))
            y = ri * (rh + 1.2)
            ax.add_patch(FancyBboxPatch((0, y + 0.4), 2.0, rh - 0.8, boxstyle="round,pad=0,rounding_size=0.4", fc=colr,
                                        ec="none"))
            ax.text(3.0, y + rh / 2, lab, va="center", fontsize=FS_S - 0.9, color=INK)
            for k, g in enumerate(top):
                sym = str(genes.loc[g, "symbol"]) or str(genes.loc[g, "gene_id"])
                hit = g in pos_rows
                is_cold = bool(cold.get((d, g), False)) if hit else False
                fc = colr if hit else "#f4f4f1"
                xx = x0 - 2 + k * (cw + 1.0)
                ax.add_patch(FancyBboxPatch((xx, y + 0.2), cw, rh - 0.4, boxstyle="round,pad=0,rounding_size=0.6",
                                            fc=fc, ec=INK if is_cold else "none", lw=0.9 if is_cold else 0))
                ax.text(xx + cw / 2, y + rh / 2, sym + ("°" if is_cold else ""), ha="center", va="center",
                        fontsize=FS_S - 1.2, color="#ffffff" if hit else INK2, fontweight="bold" if hit else "normal")
                rows.append({"disease_id": d, "method": m, "rank": k + 1, "gene_row": g, "symbol": sym,
                             "held_out_positive": hit, "reference_path_cold": is_cold})
    fig.legend(handles=[FancyBboxPatch((0, 0), 1, 1, fc=CAT[0], ec="none", label="被留出的真实关联（部署模型行）"),
                        FancyBboxPatch((0, 0), 1, 1, fc=CAT[1], ec="none", label="被留出的真实关联（仅通道行）"),
                        FancyBboxPatch((0, 0), 1, 1, fc="#ffffff", ec=INK, lw=0.9, label="° 参考路径冷的阳性"),
                        FancyBboxPatch((0, 0), 1, 1, fc="#f4f4f1", ec="none", label="未被标注为关联（不等于错误）")],
               loc="upper center", bbox_to_anchor=(0.5, 1 - 139.0 / H_), ncol=4, fontsize=FS_S - 1.2, handlelength=1.2)
    fig.text(0.5, 1 - 147.2 / H_, "选择规则（作图前写定）：在 ≥3 个留出阳性的测试疾病中，C16、C10、C18、C04 各取前 10 命中增益最大者，"
                                  "另取全体中损失最大的两个；并列时取阳性较多者。", ha="center", fontsize=FS_S - 1.4, color=INK2)
    # ---------------------------------------------------- b: gene networks of three of the cases
    from .pubfigs8x import p_network
    label_mm(fig, 2, 157, "b", "三个案例的前 10 基因网络（两种方法前 10 的并集；STRING 连接与 GO 相似度）")
    nets = [("MESH:D024741", "家族性肥厚型心肌病"), ("MESH:D000544", "阿尔茨海默病"), ("MESH:D012174", "视网膜色素变性")]
    src = {"a": pd.DataFrame(rows)}
    for k, (d, zh) in enumerate(nets):
        why = next(w for dd, w, _ in chosen if dd == d)
        nd, ed = p_network(fig, 2 + k * 60.5, 166, 59, 44, d, f"{zh}（{why.split('：')[-1]}）")
        src[f"b_{k + 1}_nodes"] = nd.assign(disease_id=d)
        src[f"b_{k + 1}_edges"] = ed.assign(disease_id=d)
    from matplotlib.lines import Line2D as _L
    fig.legend(handles=[_L([0], [0], marker="o", color="none", mfc=CAT[0], mec=SURFACE, ms=5, label="仅部署前 10"),
                        _L([0], [0], marker="o", color="none", mfc=CAT[1], mec=SURFACE, ms=5, label="仅通道前 10"),
                        _L([0], [0], marker="o", color="none", mfc="#6f6e69", mec=SURFACE, ms=5, label="两者共有"),
                        _L([0], [0], marker="o", color="none", mfc="#ffffff", mec=INK, mew=1.2, ms=5,
                           label="黑圈：留出阳性（° 冷）"),
                        _L([0], [0], color=INK2, lw=0.9, label="STRING 物理互作"),
                        _L([0], [0], color="#c9c8c2", lw=0.8, label="STRING 功能关联"),
                        _L([0], [0], color=CAT[6], lw=0.8, ls=(0, (2.2, 1.4)), label="GO Jaccard ≥ 0.2")],
               loc="upper center", bbox_to_anchor=(0.5, 1 - 213.0 / H_), ncol=7, fontsize=FS_S - 1.6,
               handlelength=1.5, columnspacing=0.8)
    fig.text(0.5, 1 - 221.6 / H_, "节点面积 ∝ log(1 + 该基因 GO 基因–术语对数)；节点内数字为该方法前 10 中的名次（共有基因为 部署/仅通道）；"
                                  "网络只用于展示，不参与任何评价或选择。", ha="center", fontsize=FS_S - 1.7, color=INK2)
    return save(fig, "Fig10_case_studies", src, {"cases": [{"disease_id": d, "why": w} for d, w, _ in chosen]})

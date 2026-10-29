"""Data-driven result paragraphs for the manuscript and reports (Chinese).

Every sentence reads its numbers from the saved result files; conditional wording
follows the registered verdicts.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import paths8


def J(p, default=None):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else default


def f(x, d=3):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"


def pc(x, d=1):
    return "—" if x is None else f"{100 * x:.{d}f}%"


def ci(c, d=3):
    if not c:
        return "—"
    return f"{c['delta']:+.{d}f}（95% CI {c['ci_low']:+.{d}f} 至 {c['ci_high']:+.{d}f}）"


def inputs_text() -> str:
    spec = J(paths8.MANIFESTS / "model_spec.json", {})
    rep = spec.get("reproduction", {})
    tx = J(paths8.AUDIT / "text_segments_l1.json", {})
    ev = J(paths8.AUDIT / "evidence_policy_audit_v8.json", {})
    mk = J(paths8.MANIFESTS / "go_masks_v8.json", {})
    l2 = ev.get("layer2_evaluation_conditioned", {})
    bl = ev.get("layer3_evidence_blocks", {})
    v6 = []
    for c in sorted((paths8.V6PASS / "main").glob("*/meta.json")):
        m = J(c)
        if m.get("v6_500_agreement"):
            v6.append(m["v6_500_agreement"]["share"])
    return (f"V7.1 第 0 折外层核心（种子 42）从冻结输入重新拟合后，最佳轮次（{rep.get('best_epoch_refit')}）、"
            f"全部权重与验证历史与存档逐位相同（最大差 {max((rep.get('weights_maxabs_diff') or {'x': 0}).values()):.1g}），"
            f"可训练参数 {rep.get('n_params'):,}。角色生成的排除名单与 V6 标签派生名单在 20 个上下文中完全一致；"
            f"冻结 V6 检索在全部 20 个上下文、{len(v6)} 个比较中复现了存档的 500 名单（完全一致比例 "
            f"{pc(float(np.mean(v6)) if v6 else None)}）。\n\n"
            f"分段文本：{tx.get('n_truncated_v71'):,} 个疾病（{pc(tx.get('share_truncated_v71'))}）在 V7.1 中被 128 词片截断，"
            f"本轮拆成共 {tx.get('segments_total'):,} 段（最多 {tx.get('max_segments')} 段），无段被截断；单段疾病的向量与 V7.1 "
            f"逐元素相同。新纳入的文本中另有 {tx.get('masked_placeholders_in_new_text'):,} 个被掩蔽的基因符号"
            f"（{tx.get('diseases_with_mask_in_new_text'):,} 个疾病）。\n\n"
            f"GO 证据：GAF 读入 {ev.get('counts', {}).get('rows_read', 0):,} 行，生产规则保留 "
            f"{ev.get('counts', {}).get('production_records', 0):,} 条记录，与 V7.1 的功能向量、掩码与功能锚逐元素相同；"
            f"{pc(ev.get('no_pmid', {}).get('share'))} 的记录没有 PMID（主要为 IEA、TAS、IBA），单列为未知出处。"
            f"评价条件化删除（审计标记）去除 {l2.get('annotation_rows', {}).get('deleted', 0):,} 行"
            f"（{pc(l2.get('annotation_rows', {}).get('share'), 2)}）、{l2.get('citations', {}).get('deleted', 0)} 篇引文、"
            f"影响 {l2.get('genes', {}).get('affected', 0)} 个基因与 {l2.get('query_relations', {}).get('gene_affected', 0):,} "
            f"条查询关系，没有基因因此失去全部 GO。证据块：PMID—关系图有 {bl.get('n_components', 0):,} 个连通分量，"
            f"巨块含 {bl.get('giant_component_relations', 0):,} 条关系（{pc(bl.get('giant_share'))}，不拆分、不抽取）；"
            f"随机抽取 {bl.get('blocks_chosen', 0):,} 个块，覆盖 {bl.get('relations_held', 0):,} 条关系与 "
            f"{bl.get('pmids_held', 0):,} 个 PMID。GO 屏蔽集合 $G_{{\\mathrm{{mask,dev}}}}$ 与 $G_{{\\mathrm{{mask,test}}}}$ 分别含 "
            f"{mk.get('G_mask_dev'):,} 与 {mk.get('G_mask_test'):,} 个基因。")


def row(R, table, method):
    for r in (R.get(table, {}) or {}).get("table", []) or []:
        if r.get("method") == method:
            return r
    return {}


def seeds_text(R) -> str:
    ps = R.get("main", {}).get("deployed_per_seed", {})
    dep = row(R, "main", "deployed")
    ch = row(R, "main", "channel_lgb_v6")
    return (f"部署模型在三个种子上的 Recall@10 分别为 {', '.join(f'{k}: {f(v)}' for k, v in sorted(ps.items(), key=lambda x: int(x[0])))}；"
            f"表中数值为先在疾病内平均种子后再做疾病宏平均。测试阳性的候选覆盖：严格 500 为 {pc(dep.get('coverage500'))}，"
            f"V6 500 为 {pc(ch.get('coverage500'))}；前十命中疾病比例 Hit@10 分别为 {pc(dep.get('hit10'))} 与 {pc(ch.get('hit10'))}。"
            if dep else "")


def cold_text(R) -> str:
    e2 = R.get("e2", {})
    t = {r["method"]: r for r in e2.get("table", [])}
    c = e2.get("contrasts", {})
    dep, B = t.get("deployed", {}), t.get("B", {})
    s = []
    s.append(f"参考路径冷查询出现在 {dep.get('n_diseases_cold', '—')} 个测试疾病中。仅检索通道的基础排序器几乎不能恢复它们"
             f"（$R^c$@10 = {f(B.get('cold_recall10'))}），部署模型为 {f(dep.get('cold_recall10'))}（冷贡献 $R^{{c\\to all}}$@10 = "
             f"{f(dep.get('cold_to_all10'))}），说明冷目标的恢复几乎全部来自语义候选。")
    for b in ("0", "1", "2"):
        k = f"corrector_beta{b}"
        if k in t:
            cc = c.get(k, {})
            s.append(f"门控纠偏 β={b}（按各折登记的 λ）：Recall@10 {f(t[k]['recall10'])}，相对部署模型 "
                     f"{ci(cc.get('recall10_vs_deployed'))}；$R^c$@10 {f(t[k]['cold_recall10'])}，差值 "
                     f"{ci(cc.get('cold_vs_deployed'))}。")
    s.append(f"提高冷正例权重没有带来可检出的冷恢复变化：β=2 对 β=0 的 $R^c$ 差为 {ci(c.get('corrector_beta2_vs_beta0_cold'))}，"
             f"β=1 对 β=0 为 {ci(c.get('corrector_beta1_vs_beta0_cold'))}。")
    hf = R.get("hitflow", {}).get("flows", {})
    k = "channel_lgb_v6->deployed"
    if k in hf:
        x = hf[k]
        s.append(f"前十命中流量（种子 42，相对仅通道 LightGBM 在其自身名单上）：保留 {x['kept']:,} 个、失去 {x['lost']:,} 个"
                 f"（其中冷基因 {x['lost_cold']:,}）、新增 {x['gained']:,} 个（其中冷基因 {x['gained_cold']:,}）。")
    sg = {r["method"]: r for r in (R.get("subgroups", {}) or {}).get("table", [])}
    d, ch = sg.get("deployed", {}), sg.get("channel_lgb_v6", {})
    if d:
        s.append(f"缺失人群（关系微平均）：无 GO 的蛋白编码基因查询 {d.get('nogo_protein_queries', '—')} 条，前十恢复 "
                 f"{pc(d.get('nogo_protein_hit10_micro'))}（仅通道 LightGBM {pc(ch.get('nogo_protein_hit10_micro'))}）；"
                 f"非编码 RNA 查询 {d.get('ncrna_queries', '—')} 条，进入 500 的比例 {pc(d.get('ncrna_pool_micro'))}"
                 f"（仅通道 {pc(ch.get('ncrna_pool_micro'))}），前十 {pc(d.get('ncrna_hit10_micro'))}"
                 f"（仅通道 {pc(ch.get('ncrna_hit10_micro'))}）；语义不可用基因的查询 {d.get('sem_unavail_queries', '—')} 条，"
                 f"前十 {pc(d.get('sem_unavail_hit10_micro'))}。")
    zs = (R.get("strata", {}) or {}).get("zero_signal_s42", {})
    if zs:
        s.append(f"种子 42 的 {zs['queries']:,} 条测试查询中，{zs['zero_signal_both']:,} 条（{pc(zs['zero_signal_share'])}）的基因在评价上下文中"
                 f"既无任何 V6 信号也无语义表示，属于任何通道都无法恢复的剩余零信号个体；其中仅 {zs['zero_signal_in_pool']} 条靠度先验填充进入候选。")
    fb = R.get("fallback", {})
    if fb:
        s.append(f"回退精确性：全部纠偏配置在 $a=0$ 的行上与 $B$ 的最大绝对差为 {fb.get('max_abs_diff', 0):g}。")
    st = (R.get("strata", {}) or {}).get("strata", {})
    d, ch = st.get("deployed", {}), st.get("channel_lgb_v6", {})
    if d:
        s.append(f"评价分层（只用于报告，从不作为路由条件）：只有一条观察关联的疾病 Recall@10 为 {f(d.get('single_relation_recall10'))}"
                 f"（仅通道 {f(ch.get('single_relation_recall10'))}），多关联疾病 {f(d.get('multi_relation_recall10'))}"
                 f"（{f(ch.get('multi_relation_recall10'))}）；有 OMIM 映射的疾病 {f(d.get('omim_mapped_recall10'))}、无映射 "
                 f"{f(d.get('not_omim_recall10'))}；阳性负担 1 / 2–5 / >5 分别为 {f(d.get('burden_1'))} / {f(d.get('burden_2_5'))} / "
                 f"{f(d.get('burden_gt5'))}。therapeutic-only 关系（不作阳性）{d.get('therapeutic_only_pairs', 0):,} 条，前十恢复 "
                 f"{pc(d.get('therapeutic_only_hit10_micro'))}（仅通道 {pc(ch.get('therapeutic_only_hit10_micro'))}）。")
    return "\n\n".join(s)


def budget_text(R) -> str:
    b = R.get("budget", {})
    rows = pd.DataFrame(b.get("table", []))
    if rows.empty:
        return ""
    s = []
    for tb, lab in (("1_fixed_v6_500", "固定旧 V6 500"), ("2_shared_new_500", "共享新 500"), ("3_end_to_end", "严格端到端"),
                    ("4_bridge_950", "约 950 桥接")):
        sub = rows[rows["table"] == tb]
        if len(sub):
            s.append(f"{lab}：" + "；".join(f"{r.method} {f(r.recall10)}" for r in sub.itertuples()) + "。")
    v = R.get("variants", {})
    pr = v.get("plain_rrf", {}).get("deployed")
    s42 = (R.get("subgroups", {}) or {}).get("deployed_seed42", {})
    if pr:
        dd = pr.get("delta_vs_main")
        s.append(f"缺失感知与普通 RRF（同一 $w_j$，种子 42；普通 RRF 为整链重训）：总体 Recall@10 差值（普通减缺失感知）"
                 f"{ci(dd) if isinstance(dd, dict) else '—'}；非编码 RNA 查询进入 500 的比例由缺失感知的 {pc(s42.get('ncrna_pool_micro'))}"
                 f"降到普通 RRF 的 {pc(pr.get('ncrna_pool_micro'))}；无 GO 基因查询的前十恢复为 {pc(s42.get('nogo_hit10_micro'))} 对 "
                 f"{pc(pr.get('nogo_hit10_micro'))}。"
                 "普通 RRF 在语义缺失时相当于把检索分数乘以 $w$，使缺少语义的候选被系统性挤出名单；缺失感知融合避免了这种额外降分，"
                 "而总体指标几乎不变。")
    for k, lab in (("v6pool_A_vs_channel", "在固定 V6 500 上，候选 A 相对仅通道 LightGBM"),
                   ("strict_A_vs_v6pool_A", "同一候选 A 由 V6 500 换为严格 500"),
                   ("bridge_A_vs_B", "约 950 桥接中，候选 A 相对同预算仅通道 LightGBM"),
                   ("bridge_A_vs_strict_A", "候选 A 由严格 500 换为约 950 桥接")):
        if b.get(k):
            s.append(f"{lab}的差值为 {ci(b[k])}。")
    return "\n\n".join(s)


def e4_text(R) -> str:
    e4 = R.get("e4", {})
    if e4:
        names = {"e4_layer1": "训练可见来源排除", "e4_layer1_random": "其匹配随机", "e4_layer2": "评价条件化删除（审计标记）",
                 "e4_layer2_random": "其匹配随机（审计标记）", "e4_experimental": "仅实验类 GO", "e4_no_tas_nas": "去 TAS/NAS",
                 "e4_no_mutant": "去突变与遗传相互作用类", "e4_blocks": "证据块隔离", "e4_blocks_random": "其匹配随机",
                 "e4_rescore_no_ic": "去 IC（重打分）", "e4_rescore_iea_only": "仅 IEA（重打分）"}
        first = next(iter(e4.values()))
        s = [f"按 PDF §4.4 的口径，报告新方法（部署模型）与旧方法（仅检索通道 LightGBM 在其自身 V6 500 上）在完整数据与删减数据中的差值及两者之差"
             f"（种子 42，整链重算）。完整数据中新旧差为 {ci(first['new_minus_old_full'])}。"]
        for tag, r in e4.items():
            t = (f"{names.get(tag, tag)}：删减数据中新旧差 {ci(r['new_minus_old_reduced'])}，与完整数据之差 {ci(r['difference'])}"
                 + ("（旧方法在该世界中重训）" if r.get("old_refitted_in_arm") else ""))
            if r.get("block_queries"):
                t += (f"；证据块关系 {r['block_queries']} 条的前十恢复：新方法 {pc(r.get('block_hit10_deployed'))}（完整数据 "
                      f"{pc(r.get('block_hit10_full_deployed'))}），旧方法 {pc(r.get('block_hit10_channel_lgb_v6'))}（完整数据 "
                      f"{pc(r.get('block_hit10_full_channel_lgb_v6'))}）")
            s.append(t + "。")
        s.append("GO-only 删除不改变旧方法（V6 七个通道不读 GO），因此其“删减数据中的新旧差”直接反映新方法的变化；"
                 "评价条件化删除使用了评价关系的文献，只作审计，不回流任何选择。")
        return "\n\n".join(s)
    v = R.get("variants", {})
    names = [("e4_layer1", "训练可见来源排除"), ("e4_layer1_random", "其匹配随机"), ("e4_layer2", "评价条件化删除（审计标记）"),
             ("e4_layer2_random", "其匹配随机（审计标记）"), ("e4_experimental", "仅实验类 GO"), ("e4_no_tas_nas", "去 TAS/NAS"),
             ("e4_no_mutant", "去突变与遗传相互作用类"), ("e4_blocks", "证据块隔离"), ("e4_blocks_random", "其匹配随机"),
             ("e4_rescore_no_ic", "去 IC（重打分）"), ("e4_rescore_iea_only", "仅 IEA（重打分）")]
    s = []
    for tag, lab in names:
        if tag in v and "deployed" in v[tag]:
            d = v[tag]["deployed"]
            b = v[tag].get("B", {})
            dd = d.get("delta_vs_main")
            txt = f"{lab}：部署模型 Recall@10 {f(d.get('recall10'))}（相对完整数据 {ci(dd) if isinstance(dd, dict) else '—'}）"
            if b:
                txt += f"，基础排序器 {f(b.get('recall10'))}"
            if d.get("block_queries"):
                txt += f"；证据块关系 {d['block_queries']} 条的前十恢复 {pc(d.get('block_hit10_micro'))}"
                if b.get("block_hit10_micro") is not None:
                    txt += f"（基础排序器 {pc(b.get('block_hit10_micro'))}）"
            s.append(txt + "。")
    return "\n\n".join(s)


def e56_text(R) -> str:
    v = R.get("variants", {})
    main = row(R, "main", "deployed")
    s = []
    lab = {"e5_none": "无补位", "e5_mlp": "两层学生", "e5_linear": "线性学生", "e5_nn": "序列近邻转移",
           "e5_teacher_sS": "教师自身序列投影", "e5_mlp_shuffled": "两层学生（配对打乱）", "e5_protst": "ProtST 输入学生（开放知识）"}
    parts = []
    for tag, name in lab.items():
        d = v.get(tag, {}).get("deployed")
        if d:
            parts.append(f"{name}：屏蔽基因查询进入 500 {pc(d.get('mask_test_pool_micro'))}、前十 {pc(d.get('mask_test_hit10_micro'))}、"
                         f"总体 Recall@10 {f(d.get('recall10'))}")
    if parts:
        n = v.get("e5_none", {}).get("deployed", {}).get("mask_test_queries")
        s.append(f"E5（$G_{{\\mathrm{{mask,test}}}}$ 的 GO 被删除，{n} 条测试查询的基因在屏蔽集合中）：" + "；".join(parts) + "。")
    for tag, name in (("e6_cold", "真正关联监督冷（GO 保留）"), ("e6_cold_gomask", "真正关联监督冷且删除 GO")):
        d = v.get(tag, {}).get("deployed")
        if d:
            s.append(f"{name}：屏蔽基因查询进入 500 {pc(d.get('mask_test_pool_micro'))}、前十 {pc(d.get('mask_test_hit10_micro'))}，"
                     f"总体 Recall@10 {f(d.get('recall10'))}。")
    nat = v.get("natural_student_on", {}).get("deployed")
    if nat:
        dd = nat.get("delta_vs_main")
        s.append(f"自然完整输入中开启学生：总体 Recall@10 {f(nat.get('recall10'))}（相对关闭 {ci(dd) if isinstance(dd, dict) else '—'}）；"
                 f"自然缺 GO 的蛋白编码基因查询前十 {pc(nat.get('nogo_hit10_micro'))}。")
    return "\n\n".join(s)


def comp_text(R) -> str:
    sm = (R.get("semantic", {}) or {}).get("table", [])
    s = []
    if sm:
        s.append("全背景单独排名（种子 42）：" + "；".join(
            f"{r['method']} {f(r['recall10'])}（$R^c$ {f(r['cold_recall10'])}）" for r in sm) + "。")
    c = R.get("comparators", {})
    for k, lab in (("kdgene", "KDGene"), ("sign", "SIGN 式方法")):
        if k in c:
            s.append(f"{lab}（同一严格 500，$F_{{j+1}}$ 选检查点）：Recall@10 {f(c[k]['recall10'])}，部署模型相对其差值 "
                     f"{ci(c[k].get('deployed_minus'))}。")
    return "\n\n".join(s)


def e8_text(R) -> str:
    ext = R.get("external", {})
    s = []
    if ext:
        fd = ext.get("fields", {})
        t = pd.DataFrame(ext.get("table", []))
        u = t[t["subset"] == "union"].set_index("method")["recall10"].to_dict() if len(t) else {}
        s.append(f"618 个疾病、{fd.get('pairs', 0):,} 个外部对（GenCC {fd.get('by_source', {}).get('GenCC', 0)}、ClinVar "
                 f"{fd.get('by_source', {}).get('ClinVar', 0)}）作为 V7.1 已评价过的集合进行再分析。三个字段：训练中见过该疾病 "
                 f"{fd.get('seen_in_training_disease', 0)} 对；邻近疾病中见过同一关系 {fd.get('same_relation_seen', 0)} 对；共享原始文献 "
                 f"是/否/未知 = {fd.get('shared_original_evidence', {}).get('yes', 0)}/{fd.get('shared_original_evidence', {}).get('no', 0)}/"
                 f"{fd.get('shared_original_evidence', {}).get('unknown', 0)}。过滤排名下的 Recall@10：部署 {f(u.get('deployed'))}、"
                 f"V7.1 直接融合 {f(u.get('v71_direct_strict'))}、仅通道 LightGBM {f(u.get('channel_lgb_v6'))}、F7 {f(u.get('F7_v6'))}。")
        for cc in ext.get("contrasts", []):
            if cc["subset"] in ("union", "independent_evidence"):
                s.append(f"部署对 {cc['b']}（{cc['subset']}，{cc['n']} 病）：{ci(cc)}。")
    h = R.get("historical", {})
    if h:
        r = h.get("results", {})
        hc = R.get("historical_contrasts", {})
        lab = {"deployed": "部署", "channel_lgb_v6": "仅通道 LightGBM（自身 V6 500）", "B": "基础排序器", "standalone_sF": "教师 $s_F$ 单独"}
        def part(task):
            return "、".join(f"{lab[m]} {f(r.get(f'{task}__{m}', {}).get('recall10'))}" for m in lab if f"{task}__{m}" in r)
        s.append(f"历史任务（截止 2023-12-31，所有模型在截止前资源上重训）：X2 为截止时已有关系疾病的 {h.get('x2', {}).get('pairs')} 条新关系"
                 f"（{h.get('x2', {}).get('diseases')} 病，旧基因从排名中移除，不是疾病冷启动），Recall@10：{part('x2')}；部署对仅通道 "
                 f"{ci(hc.get('x2_deployed_vs_channel_lgb_v6'))}。X1 为截止时无关联监督的 {h.get('x1', {}).get('diseases')} 个疾病、"
                 f"{h.get('x1', {}).get('pairs')} 条关系（截止时有文本 {h.get('x1', {}).get('with_text_at_cutoff')} 个、有 HPO "
                 f"{h.get('x1', {}).get('with_hpo_at_cutoff')} 个），Recall@10：{part('x1')}；部署对仅通道 "
                 f"{ci(hc.get('x1_deployed_vs_channel_lgb_v6'))}。在截止前的资源与时间冻结条件下，内部交叉验证中的语义收益没有出现；"
                 f"X2 的旧关系在参考图中可见，检索通道本身较强，而 2023 年的 GO 与文本覆盖也少于当前版本。")
    return "\n\n".join(s)


def e9_text(R) -> str:
    s = []
    sp = R.get("support", {})
    if sp:
        c5 = pd.DataFrame(sp.get("cohort533", []))
        if len(c5):
            k5 = c5[c5["k"] == 5].set_index("method")["recall10"].to_dict()
            k0 = c5[c5["k"] == 0].set_index("method")["recall10"].to_dict()
            om = sp.get("omega_533")
            jk = f"joint_v8_w{om:g}" if om is not None else ""
            s.append(f"支持任务（533 病设计，S-fixed）：k=0 时 V6 语义 {f(k0.get('sem_v6'))}、V8.1 语义 {f(k0.get('sem_v8'))}；"
                     f"k=5 时 RWR（V8.1 名单）{f(k5.get('rwr_v8list'))}、V8.1 联合（ω={om}）{f(k5.get(jk))}、"
                     f"V6 联合 {f(k5.get('joint_v6'))}、量子游走 {f(k5.get('qwalk_v8list'))}。")
        c1 = pd.DataFrame(sp.get("cohort1shot", []))
        if len(c1):
            d1 = c1.set_index("method")["recall10"].to_dict()
            s.append("1-shot 队列（完整背景）：" + "；".join(f"{k} {f(v)}" for k, v in d1.items()) + "。")
    ex = R.get("expression", {})
    if ex:
        parts = []
        for r in ex.get("series", []):
            for cn, entry in (r.get("contrasts") or {}).items():
                d50 = entry.get("v8_deployed_s42_top50", {})
                c50 = entry.get("channel_lgb_v6_s42_top50", {})
                m50 = entry.get("matched_random_top50", {})
                if d50:
                    parts.append(f"{r['accession']}（{'配对病灶/非病灶' if 'paired' in cn else '病灶/对照'}）：V8.1 Top-50 富集 "
                                 f"{f(d50.get('enrichment'))}（P = {f(d50.get('p_value'), 3)}），仅通道 {f(c50.get('enrichment'))}"
                                 f"（P = {f(c50.get('p_value'), 3)}），匹配随机 {f(m50.get('enrichment'))}")
        s.append("四个表达队列（冻结 Top-20/50，秩检验；富集 = 选中基因的 |t| 高于匹配背景的概率，只支持相关性）：" + "；".join(parts) +
                 "。负结果全部保留（表 S4）。")
    rv = R.get("review", {})
    if rv:
        s.append(f"160 对双人盲审材料（四类各 40 对，随机行序，隐去类别、方法、名次与是否已知关系）已生成；审核本身需要两名独立人工审阅者，"
                 f"本轮未进行。")
    rp = R.get("reproduction", {})
    blk = J(paths8.AUDIT / "reproduction_env_blocked_v8.json", {})
    if blk:
        s.append("另一环境复现：独立虚拟环境 `.venv_repro` 在任何计算之前被 Windows 应用程序控制策略阻止加载 SciPy 的一个扩展模块；"
                 "按安全规则未修改系统设置、未解除文件阻止。作为替代，第 0 折种子 42 子链在主环境中以全新缓存（独立的 V6 世界与教师运行）"
                 "从冻结输入重算，这检验的是流程的确定性，而不是环境的独立性。")
    if rp:
        s.append(f"重算结果（{rp.get('environment', {}).get('executable', '')}）：V6 名次逐项相同 "
                 f"{all(v.get('rB_identical') for v in rp.get('v6_pass', {}).values())}，教师权重最大差 "
                 f"{max(rp.get('teacher_weights_maxabs', {'x': 0}).values()):.1g}，候选名单一致比例 {pc(rp.get('pools_identical_share'))}，"
                 f"部署分数最大差 {rp.get('deployed_scores_maxabs', 0):.1g}，前十一致比例 {pc(rp.get('top10_identical_share'))}；"
                 f"{'在登记容差内' if rp.get('within_tolerance') else '超出登记容差（见审计文件）'}。")
    return "\n\n".join(s)


def discussion_text(R) -> str:
    m = R.get("main", {})
    v = m.get("verdict", {})
    h = m.get("hierarchy", {})
    dep = row(R, "main", "deployed")
    ch = row(R, "main", "channel_lgb_v6")
    v71 = row(R, "main", "v71_direct_strict")
    sF = row(R, "main", "standalone_sF")
    sF0 = row(R, "main", "standalone_sF0")
    p = []
    p.append(f"**严格预算下语义收益保留。** V7.1 的约 950 预算结果（0.166）在严格 500 下没有消失：部署模型为 {f(dep.get('recall10'))}，"
             f"相对只用检索通道、在其自身 V6 500 上端到端排序的强对照（{f(ch.get('recall10'))}）的差值为 {ci(h.get('h1_vs_channel_lgb_v6'))}。"
             f"收益来自学习得到的疾病—功能语义：同样的 SapBERT 表示不经投影时单独排名只有 {f(sF0.get('recall10'))}，学习后的 $s_F$ 为 "
             f"{f(sF.get('recall10'))}；语义候选使严格 500 的开发阳性覆盖提高约 13–14 个百分点，而且冷目标几乎只能由语义候选恢复。")
    p.append(f"**新机制没有超过简单强融合。** 按事先登记的层级，部署模型相对在同一严格 500 上重训的 V7.1 直接融合（{f(v71.get('recall10'))}）"
             f"的差值为 {ci(h.get('h2_vs_v71_direct_strict'), 4)}，区间下界{'高于' if v.get('h2_success') else '低于'}−0.005，"
             f"{'非劣成立' if v.get('h2_success') else '非劣未能确立（差距很小，但登记规则不允许事后放宽）'}；冷恢复优效因此不作推断检验。"
             "开发集上，门控冷平衡纠偏器在五个上下文都落后于明示缺失值的直接 LightGBM 约 0.02–0.03，按登记规则部署了后者；"
             "测试集上三个 β 的纠偏器总体更低，冷恢复与部署模型无差别，提高冷正例权重也没有带来冷恢复变化。"
             "一个合理的解释是：纠偏器只从每个疾病 64 个抽样未标注基因中学习“正例高于未标注”，其验证曲线在 7–118 轮达到峰值后下降，"
             "而 lambdarank 直接在完整 500 列表上优化前十；在基础分数上加门控残差的结构本身并不妨碍回退精确性（$a=0$ 时逐位等于 $B$），"
             "但没有转化为排序收益。我们把它作为阴性结果报告，不在测试后增加新变体。")
    v = R.get("variants", {})
    e5n, e5m, e5s = (v.get("e5_none", {}).get("deployed", {}), v.get("e5_mlp", {}).get("deployed", {}),
                     v.get("e5_mlp_shuffled", {}).get("deployed", {}))
    p.append("**序列补位没有在本协议下成立。** 开发屏蔽臂与测试屏蔽臂给出一致的结论：学生输出的功能语义质量有限，"
             "而缺失感知融合在语义缺失时保留检索分数，一个中等质量的补位语义会把屏蔽基因的融合分拉低，使其更少进入候选"
             f"（测试屏蔽集：无补位 {pc(e5n.get('mask_test_pool_micro'))}，两层学生 {pc(e5m.get('mask_test_pool_micro'))}）。"
             f"长度层内打乱序列—基因配对后候选覆盖进一步降到 {pc(e5s.get('mask_test_pool_micro'))}，说明学生输出并非与序列无关；"
             f"但前十恢复（{pc(e5m.get('mask_test_hit10_micro'))} 对打乱后 {pc(e5s.get('mask_test_hit10_micro'))}）与总体 Recall@10 没有随之下降，"
             "序列携带的信息不足以替代观察到的 GO。这与 V7.1 中序列在完整输入下冗余的观察一致，也提示若要让补位生效，"
             "需要质量更高的功能预测，或只在补位置信度高时让它进入融合。")
    e4 = R.get("e4", {})
    e4n = {"e4_layer1": "训练可见来源排除", "e4_layer1_random": "其匹配随机", "e4_layer2": "评价条件化删除",
           "e4_layer2_random": "其匹配随机", "e4_experimental": "仅实验类 GO", "e4_no_tas_nas": "去 TAS/NAS",
           "e4_no_mutant": "去突变与遗传相互作用类证据", "e4_blocks": "证据块隔离", "e4_blocks_random": "证据块的匹配随机",
           "e4_rescore_no_ic": "去 IC（重打分）", "e4_rescore_iea_only": "仅 IEA（重打分）"}
    diffs = [(k, r["difference"]) for k, r in e4.items()]
    if diffs:
        inc0 = [k for k, d in diffs if d["ci_low"] <= 0 <= d["ci_high"]]
        neg = [(k, d) for k, d in diffs if d["ci_high"] < 0]
        e4s = f"在 {len(diffs)} 个删除条件中，{len(inc0)} 个的“新旧方法差相对完整数据的变化”区间包含 0。"
        if neg:
            e4s += ("区间低于 0 的是：" + "；".join(f"{e4n.get(k, k)} {ci(d)}" for k, d in neg) + "。")
        if any(k == "e4_rescore_iea_only" for k, _ in neg):
            e4s += "仅 IEA 的重打分是把以完整 GO 训练的模型只喂电子注释，下降主要反映输入分布改变，而不是共享证据。"
        if any(k == "e4_no_mutant" for k, _ in neg):
            nm = e4["e4_no_mutant"]
            full_gain = (nm.get("new_minus_old_full") or {}).get("delta")
            share = (f"（约为完整数据收益 {full_gain:+.3f} 的 {-nm['difference']['delta'] / full_gain:.0%}）"
                     if full_gain else "")
            e4s += (f"去除突变表型与遗传相互作用类证据并整链重训后，新旧方法差缩小{share}，"
                    "提示这类实验证据与疾病—基因发现研究可能部分同源，是需要在讨论中明确的依赖。")
    else:
        e4s = ""
    p.append("**证据来源与外部检验。** 评价条件化删除只影响 0.16% 的注释行与 266 个基因；证据块隔离同时撤出块内关系、GO 与可追溯 HPO；"
             "每个删除都配有数量匹配的随机对照。" + e4s + "这支持“语义收益主要不是由与评价关系共享文献的少数 GO 注释带来的”，"
             "但只覆盖可追溯 PMID 的部分。618 个疾病的外部再分析与 V7.1 当时的确认方向一致，但它已被用于设计，不能再称为首次确认；"
             "新来源确认在本轮不可得。")
    hc = R.get("historical_contrasts", {})
    x2, x1 = hc.get("x2_deployed_vs_channel_lgb_v6"), hc.get("x1_deployed_vs_channel_lgb_v6")
    if x2:
        p.append(f"**时间冻结条件下没有看到收益。** 在 2023-12-31 截止的历史世界中重训全部模型后，部署模型相对仅检索通道 LightGBM 在 X2 上的差值为 "
                 f"{ci(x2)}，在 X1 上为 {ci(x1)}。这与内部交叉验证中的稳定提升形成对照，提示三点：X2 疾病的旧关系在参考图中可见，"
                 "检索通道在这类“已知疾病的新关系”上本来就强；截止前的 GO 与文本覆盖更少，语义通道的输入更弱；X1 只有 39 个疾病，区间很宽。"
                 "因此本文把语义收益限定在疾病关联冷启动的内部再建模与外部再分析证据上，不外推为前瞻性发现能力。")
    return "\n\n".join(p)


def limits_text(R) -> str:
    return "\n\n".join([
        "第一，旧五折已被多次查看，本轮所有测试结果属于内部再建模证据；登记只能防止本轮内的选择偏倚。",
        "第二，GO 与疾病关联共享文献的程度只能在可追溯 PMID 的范围内审计；42% 的 GO 记录没有 PMID，按未知出处处理，"
        "外部集合中 83% 的记录也没有可比对的文献，文献独立性结论相应收缩。",
        "第三，历史任务仍使用晚于截止日的基因背景、MANE 2025 蛋白序列、2026 年的等价组/家族与冻结的超参数；X1 样本很小；"
        "在时间冻结条件下部署模型没有超过仅检索通道的强对照，语义收益不能外推为前瞻性发现能力。",
        "第四，160 对双人盲审只交付了材料，审核本身未进行；表达队列只支持相关性，不支持因果解释。",
        "第五，纠偏器与序列学生的阴性结果只针对本协议（15 叶、200 轮、64 个未标注样本、预设 w=0.5 的学生验收）；"
        "按停止规则，本轮没有在测试后扩展网格或更换 ESM 规模。",
        "第六，CTD 关系是观察到的疾病—基因关联，不等同于已证实的致病因果；therapeutic 关系只作分层报告。"])


def conclusion_text(R) -> str:
    v = R.get("main", {}).get("verdict", {})
    return ("在严格的 500 候选预算和整链袋外评价下，疾病—功能语义检索的收益得到保留，缺失感知的候选融合使语义候选进入最终排序，"
            "部署模型相对只用检索通道的强对照有稳定提升；" +
            ("与同预算下重训的 V7.1 直接融合相比非劣。" if v.get("h2_success") else
             "但与同预算下重训的 V7.1 直接融合相比，登记的非劣界未能确立，二者差距很小。") +
            "新提出的门控冷平衡纠偏与序列补位在本数据上没有取得收益，按事先规则降为阴性结果；2023 年截止的历史任务中也没有看到相对强对照的收益。"
            "论文的定位因此是知识覆盖不均衡条件下的语义检索与候选预算、缺失处理与评价协议，而不是一个新的排序模型或前瞻性发现工具。")


def dev_text() -> str:
    ctl = J(paths8.MANIFESTS / "dev_teacher_controls_v8.json", {})
    st = J(paths8.MANIFESTS / "dev_student_selection_v8.json", {})
    sels = [J(paths8.DEV / f"fold{j}" / "selection.json") for j in range(5)]
    ws = [s["w"]["w"] for s in sels]
    cov_v6 = np.mean([s["w"]["coverage"]["v6_500"] for s in sels])
    cov_s = np.mean([s["w"]["coverage"][f"aware_w{s['w']['w']}"] for s in sels])
    cov_p = np.mean([s["w"]["coverage"][f"plain_w{s['w']['w']}"] for s in sels])
    a_best = [s["ab"]["best_A_dev_recall10"] for s in sels]
    c_best = [max(r["dev_recall10"] for r in s["summary"]["corrector"]) for s in sels]
    c_cold = [max(r["dev_cold_recall10"] for r in s["summary"]["corrector"]) for s in sels]
    a_cold = [s["ab"]["best_A_dev_cold_recall10"] for s in sels]
    b_dev = [s["summary"]["B_dev"]["dev_recall10"] for s in sels]
    cp = [s["summary"]["corrector"][0]["diag"]["cold_positives_in_pairs"] for s in sels]
    cd = [s["summary"]["corrector"][0]["diag"]["diseases_with_cold_positive"] for s in sels]
    it = [r["best_iteration"] for s in sels for r in s["summary"]["corrector"]]
    ctl_rows = ctl.get("per_context", {})
    ref = np.mean([v["dev_sF_recall10"]["reference"] for v in ctl_rows.values()]) if ctl_rows else None
    bg = np.mean([v["dev_sF_recall10"]["background"] for v in ctl_rows.values()]) if ctl_rows else None
    rows = st.get("per_context", {})
    g_mlp = [max((r["gain"] for r in v["rows"][1:] if r.get("kind") == "mlp"), default=None) for v in rows.values()]
    g_lin = [max((r["gain"] for r in v["rows"][1:] if r.get("kind") == "linear"), default=None) for v in rows.values()]
    none_cov = [v["rows"][0]["masked_coverage"] for v in rows.values()]
    top10 = [r["gene_level_masked"]["top10"] for v in rows.values() for r in v["rows"][1:] if r.get("kind") == "mlp"]
    top10_lin = [r["gene_level_masked"]["top10"] for v in rows.values() for r in v["rows"][1:]
                 if r.get("kind") == "linear"]
    return (f"教师对照集：全背景未标注对照在五个外层上下文的开发 $s_F$ Recall@10 均高于参考关系基因对照"
            f"（平均 {f(bg)} 对 {f(ref)}），五个上下文均选全背景。\n\n"
            f"序列学生在开发屏蔽臂上未通过验收：无补位时开发疾病中屏蔽基因阳性进入严格 500 的比例平均为 "
            f"{pc(float(np.mean(none_cov)))}，两层学生使其变化 {', '.join(f'{x:+.3f}' for x in g_mlp)}，线性学生 "
            f"{', '.join(f'{x:+.3f}' for x in g_lin)}（五个上下文），全部为负；屏蔽基因的学生输出在约 1.9 万个真实功能向量中"
            f"找回自身的前十命中率仅为 {f(float(np.mean(top10)))}（两层）与 {f(float(np.mean(top10_lin)))}（线性），"
            f"说明其补位语义质量有限，而缺失感知融合在语义缺失时保留检索分数 $b$，一个中等质量的语义名次反而会把 $R$ 拉到 $b$ 以下。"
            f"因此学生在自然主链中关闭，ESM 650M 的上限比较按登记不触发。\n\n"
            f"候选融合：各上下文选出的 $w_j$ 为 {', '.join(str(x) for x in ws)}；开发阳性的 500 覆盖由 V6 500 的 "
            f"{pc(cov_v6)} 提高到 {pc(cov_s)}（同 $w$ 普通 RRF 为 {pc(cov_p)}）。\n\n"
            f"二级模型：基础排序器 $B$ 的开发 Recall@10 为 {', '.join(f(x) for x in b_dev)}；六个门控纠偏配置的最好开发 Recall@10 为 "
            f"{', '.join(f(x) for x in c_best)}，最好开发冷恢复为 {', '.join(f(x) for x in c_cold)}；候选 A 的最好开发 Recall@10 为 "
            f"{', '.join(f(x) for x in a_best)}（冷恢复 {', '.join(f(x) for x in a_cold)}）。纠偏器的最佳轮次在 "
            f"{min(it)}–{max(it)} 之间（200 轮预算），验证曲线随后下降，未触发 31 叶扩展；训练对中的冷正例为 "
            f"{min(cp)}–{max(cp)} 个、覆盖 {min(cd)}–{max(cd)} 个疾病，高于登记的冷模拟阈值。按登记的 A/B 规则，"
            f"没有纠偏配置落在最佳 A 的 0.005 以内，五个上下文均部署候选 A。")

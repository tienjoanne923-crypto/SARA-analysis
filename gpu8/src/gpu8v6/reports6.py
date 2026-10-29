"""Standalone V6 reports: execution/results summary, closure table, not-done list,
and the delivery note next to the manuscript.  Numbers come from the collected
numbers file so the reports and the manuscript cannot drift apart."""
from __future__ import annotations

import json

from gpu8 import paths as p5

from . import paths6
from .appendix6 import closure_rows
from .common import now_utc
from .manuscript6 import OUT_DIR, TITLE


def f(v, k, d=3):
    return f"{v[k]:.{d}f}"


def pc(v, k, d=1):
    return f"{100 * v[k]:.1f}%"


def ci(v, pre, d=4):
    return (f"{v[pre + '_delta']:+.{d}f}（95% CI {v[pre + '_ci_low']:+.{d}f} 至 "
            f"{v[pre + '_ci_high']:+.{d}f}）")


def summary(v: dict) -> str:
    L = [f"# V6 执行与结果报告", "", f"生成时间：{now_utc()}（依据 DiseaseGene_V6_Remodel_RepairPlan.pdf）", "",
         "## 一句话结论", "",
         (f"V6 把共享 500 候选的 oracle 上界从 {f(v,'z_v5_b3_cpu_fusion_oracle_recall10')} 提到 "
          f"{f(v,'z_v6_lightgbm_oracle_recall10')}，端到端 Recall@10 从 {f(v,'z_v5_b3_cpu_fusion_recall10_end',4)} 提到 "
          f"{f(v,'z_v6_lightgbm_recall10_end',4)}（登记检验 Z-A {ci(v,'cmp_ZA')}，Holm p<0.001）；收益来自候选恢复与合法新数据，"
          f"学习型排序器相对调优融合无显著差异（Z-B {ci(v,'cmp_ZB')}）。支持驱动传播在固定候选上显著有效（S-A {ci(v,'s_SA')}，"
          f"Holm p {f(v,'s_SA_holm_p')}），来源组加权无收益（S-C {ci(v,'s_SC')}），按停止规则撤回证据依赖提升准确率的主张。"),
         "", "## 按方案章节的执行结果", ""]
    rows = [
        ("§2.1/§4.3 基因 HPO 安全重建", "来源排除后聚合；严格/宽松两版", f"留一覆盖贡献 {f(v,'src_gene_hpo_coverage_contribution')}"),
        ("§2.2/§4.2 GO/通路/表达", f"{int(v['gene_feat_dim'])} 维特征 + 缺失标志；功能桥接覆盖 {int(v['bridge_genes'])} 个基因",
         f"评分器中加 GO {f(v,'gab_inc_plus_go',4)}（阴性）；桥接用于传播 +{f(v,'s2_mix_vs_ppi_rwr_equal_k5_delta',4)}"),
        ("§2.5 消融差值方向", "统一 Δabl = Mabl − Mfull，配对种子，候选哈希",
         f"A1 {f(v,'abl_a1_delta_abl',4)}，A2 {f(v,'abl_a2_delta_abl',4)}，A3 {f(v,'abl_a3_delta_abl',4)}（候选改变），A4 {f(v,'abl_a4_delta_abl',4)}；V5 表 4 种子不配对"),
        ("§2.6 当前支持的证据组", "只在显露支持内建组",
         f"k=5 种子改变 {pc(v,'fl_k5_layer1_share_seed_changed')}，前十改变 {pc(v,'fl_k5_layer3_share_top10_changed')}，指标改变 {pc(v,'fl_k5_layer4_share_metric_changed')}"),
        ("§2.7 冷验证曲线", "KDGene 100-epoch 预算、耐心 10",
         f"最佳 epoch {int(v['kd_best_epoch'])}，冷验证 {f(v,'kd_best_val',4)}；损失 {f(v,'kd_loss_first')}→{f(v,'kd_loss_last')} 仍在下降"),
        ("§3.2 oracle 分解", "R = U·η 逐病恒等", f"V5 U@10 {f(v,'v5_b3_U')}，η {f(v,'v5_b3_eta')}"),
        ("§3.3 500→1000 命中流", "嵌套与分数一致性核对",
         f"新增基因占前十 {int(v['hf_total_top_slots_taken_by_new'])} 位，其中阳性 {int(v['hf_total_new_positive_entered_top'])}；挤出原命中 {int(v['hf_total_pushed_out'])}"),
        ("§4.1 可用性审计", "三类无 HPO；映射修复；交叉表",
         f"HPO {pc(v,'hpo_v5')}→{pc(v,'hpo_v6')}；5-shot 无 HPO {pc(v,'five_nohpo_v5')}→{pc(v,'five_nohpo_v6')}（资源无记录 {int(v['five_no_record'])}）"),
        ("§4.4 非 HPO 检索", "掩蔽文本 TF-IDF；IC 加权本体",
         f"文本留一贡献 {f(v,'src_text_coverage_contribution')}，本体 {f(v,'src_onto_coverage_contribution')}"),
        ("§5 分流", "只用可观测信息", f"Z1 {int(v['route_Z1_hpo'])} / Z2 {int(v['route_Z2_no_hpo_text_or_ontology'])} / Z3 {int(v['route_Z3_no_legal_disease_information'])}"),
        ("§6 零支持检索", "开发集冻结倒数排名融合；预算 500；回退原因",
         f"五折覆盖 {f(v,'z_v5_b3_cpu_fusion_coverage')}→{f(v,'test_cov')}；零覆盖 {pc(v,'z_v5_b3_cpu_fusion_zero_coverage')}→{pc(v,'z_v6_lightgbm_zero_coverage')}；救回 {int(v['rescued_n'])}/{int(v['v5_zero_n'])}"),
        ("§7 评分阶段", "35/210 点单纯形 + 细化；LR/LightGBM/κ·tanh 残差；1-SE",
         f"F7 {f(v,'z_v6_F7_recall10_end',4)}，LightGBM {f(v,'z_v6_lightgbm_recall10_end',4)}，残差不准入"),
        ("§8 疾病置换", "P1/P2/P3", f"V5 b9 P1 {f(v,'pv5_P1_distant_delta',4)}、P2 {f(v,'pv5_P2_distant_delta',4)}；V6 LightGBM P2 {f(v,'pv6_lightgbm_P2_distant_all_delta',4)}；注意力 P1 {f(v,'pv6_attention_residual_P1_distant_all_delta',4)}"),
        ("§9 支持吸收", "残差停止 RWR、P_F、P_mix；S-fixed/S-adaptive；联合 ω",
         f"k=1/3/5：{f(v,'dose_rwr_equal_S_fixed_k1',4)}/{f(v,'dose_rwr_equal_S_fixed_k3',4)}/{f(v,'dose_rwr_equal_S_fixed_k5',4)}（语义 {f(v,'dose_sem_S_fixed_k0',4)}）；重连后 k=5 {f(v,'rw_k5_rewired',4)}"),
        ("§10 证据依赖", "四层诊断、Δseed、Γgrp、连接度权重", f"Γgrp k=5 {ci(v,'s2_Gamma_grp_k5')}"),
        ("§11 模型选择", "开发/内层选择；登记后评价；1-SE", f"开发拟合 {int(v['devfit_n'])} 次；正式学习型拟合 {int(v['n_formal_learned'])} 次（≤45）"),
        ("§12.1 时间外", "全输入历史审计；存档头部核验；X2 支持传播",
         f"传播 − 2023 先验 Recall@50 {ci(v,'tc_rwr_equal_mix0.3__minus__degree_prior_2023',3)}"),
        ("§12.2 ClinVar", "变异→条件→文献", f"核实独立 {int(v['cv_independent_verified'])}/{int(v['cv_pairs'])}，未知率 {pc(v,'cv_unknown_rate')}"),
        ("§12.3 外部", "分层分母；中等证据单列",
         f"GenCC 确定/强 n={int(v['ext_gencc_definitive_strong_n'])}：F7 {f(v,'ext_gencc_definitive_strong_F_main_R')}；ClinVar≥2星 n={int(v['ext_clinvar_2star_n'])}：F7 {f(v,'ext_clinvar_2star_F_main_R')}"),
        ("§12.4 验证链", "保度重连；真实 Top-20/50 表达；盲审",
         f"GSE13355 前 20 AUC {f(v,'ex_GSE13355_lesional_vs_uninvolved_paired_v6_main_lightgbm_top20_auc')}（未复现）；盲审未完成"),
        ("§13 工程", "单运行权威结果、原子写、内容哈希、失败保留", f"{int(v['n_runs_v6'])} 次运行；失败尝试 {int(v['n_failed_attempts'])} 次（已记录）"),
        ("§17 论文重写", "新题目、新结构、统一差值方向、OMML 公式", "见 manuscript_v6/"),
    ]
    L.append("| 方案章节 | 做了什么 | 关键结果 |")
    L.append("|---|---|---|")
    for a, b, c in rows:
        L.append(f"| {a} | {b} | {c} |")
    L += ["", "## 登记检验（Holm 在族内）", "",
          "| 比较 | 差值（95% CI） | Holm p |", "|---|---|---|",
          f"| Z-A V6 管线 − V5 融合（Recall@10） | {ci(v,'cmp_ZA')} | <0.001 |",
          f"| Z-B LightGBM − F7（Recall@10） | {ci(v,'cmp_ZB')} | {f(v,'cmp_ZB_holm_p')} |",
          f"| Z-C V6 − V5 检索（U@10） | {ci(v,'cmp_ZC')} | <0.001 |",
          f"| S-A 等权 RWR k=5 − 语义 k=0 | {ci(v,'s_SA')} | {f(v,'s_SA_holm_p')} |",
          f"| S-B 联合 − 等权 RWR k=5 | {ci(v,'s_SB')} | {f(v,'s_SB_holm_p')} |",
          f"| S-C 组种子 − 等权 k=5 | {ci(v,'s_SC')} | {f(v,'s_SC_holm_p')} |", "",
          "## 保留的阴性结果", "",
          "- 64 维元岭表示空间适配：已撤下主路径（V5 结论，V6 未重跑）。",
          f"- 神经残差：开发拟合 {pc(v,'devfit_kappa0_share')} 选 κ=0；正式拟合测试无增益；P1 ≈ 0；不准入。",
          f"- 基因功能特征用于评分器：加 GO {f(v,'gab_inc_plus_go',4)}，通路与表达无变化。",
          f"- 来源组加权：S-C {ci(v,'s_SC')}；Γgrp k=5 {ci(v,'s2_Gamma_grp_k5')}。",
          f"- 支持驱动的候选再生成：S-adaptive − S-fixed（k=5）{ci(v,'s2_adaptive_vs_fixed_joint_w0.75_k5')}。",
          f"- 时间任务：传播未稳定超过 2023 度先验（Recall@50 差 {ci(v,'tc_rwr_equal_mix0.3__minus__degree_prior_2023',3)}）。",
          f"- KDGene：合法冷验证早停后第 0 折 {f(v,'kd_test_R',4)}，仍最弱。", ""]
    return "\n".join(L)


def not_done() -> str:
    return "\n".join([
        "# V6 未完成与受限项", "",
        "1. 160 对双人盲审：只交付去标识抽样表（V5 已生成），实际人工审阅与分歧处理未完成；任何文稿都不得把抽样表计为已完成核查。",
        "2. 真正未参与设计的新外部来源：没有。外部结果是对已查看外部集的探索性再分析；GenCC 确定/强层仅 20 个疾病。",
        "3. 时间任务：2.66 GB 历史关系原始转储未重下，仅核验 Wayback 存档头部（Report created: Wed Nov 29 2023）与 CDX 索引；使用处理后的 2023Q4 边表；Mondo 无 2023 发布可用，时间任务不用 Mondo。",
        "4. ClinVar：疾病条件核实为变异级唯一性规则；SCV 级核实需 XML 发布，未做；未知率 45.4%。",
        "5. 比较器：SIGN 式对照未在 V6 候选上重新调参；KDGene 冷验证曲线只在第 0 折以开发预算运行（未做五折三种子）。",
        "6. 学习式检索器（方案 6.3 可选第三条）：未实现。",
        "7. 五折结果的性质：V5 已使用这五折，V6 结果为内部估计，不是从未参与设计的确认性测试。",
        "8. 表达支持：只有 GSE13355 出现阳性信号，GSE30999 与两个 UC 队列未复现，未做多重比较校正。",
        "9. 一次正式运行（V6Z__lightgbm__f0__s42）首次尝试在保存模型时失败（非 ASCII 路径），已重跑；失败记录见 manifests_v6/failed_attempts_v6.json。",
        ""])


def delivery(v: dict) -> str:
    md = f"{TITLE}_V6_20260923.md"
    return "\n".join([
        "# 交付说明（第六轮 V6）", "",
        f"生成时间：{now_utc()}", "",
        "## 文件", "",
        f"- 稿件：`manuscript_v6/{md}` 与同名 `.docx`（公式为 Word 原生 OMML 对象；图 8 幅，表 11 张）。",
        "- 图：`gpu8/results_v6/figures/`（6.75 in、600 dpi、灰度、SimSun，PNG/SVG/PDF）。",
        "- 代码：`gpu8/src/gpu8v6/`（与 V5 的 `gpu8/src/gpu8/` 分开，V5 代码哈希与结果不变）；测试 `gpu8/tests/test_v6_*.py`。",
        "- 结果：`gpu8/results_v6/`（audit、retrieval、scoring、support、permutation、external、tables）；逐运行权威结果 `gpu8/runs_v6/*/result.json`。",
        "- 冻结与登记：`gpu8/manifests_v6/`（retriever_v6_frozen、scorer_v6_frozen、devfit_v6_selection、z_main_scorer、support_v6_frozen、protocol_v6_registered、run_ledger_v6、failed_attempts_v6）。",
        "- 报告：`gpu8/reports_v6/`（本执行报告、关闭表、未完成项、稿件数值溯源 manuscript_numbers_v6.csv）。",
        "- 新下载资源：`gpu8/external/v6/`（GTEx v8、Reactome 当前/86、GO 当前/2023-10-09、GOA 人类 GAF 当前/2023-10-09、HPO genes_to_phenotype 2023-10-09、HGNC 2023-10-01、ClinVar var_citations、CTD Wayback CDX 与存档头部 256 KB），逐项 SHA-256 见 download_log_v6.csv。",
        "",
        "## 复现顺序（Windows PowerShell）", "",
        "```",
        "$env:PYTHONPATH='D:\\信\\gpu8\\src'",
        "python -m gpu8v6.audit_metrics; python -m gpu8v6.hitflow; python -m gpu8v6.phenotypes_v6",
        "python -m gpu8v6.disease_text; python -m gpu8v6.gene_features --version current",
        "python -m gpu8v6.availability; python -m gpu8v6.disease_semantics; python -m gpu8v6.tune_retrieval",
        "python -m gpu8v6.build_candidates; python -m gpu8v6.zero_support --stage tune",
        "python -m gpu8v6.zero_support --stage devfit; python -m gpu8v6.select_main; python -m gpu8v6.support_line --stage select",
        "python -m gpu8v6.register_protocol",
        "python -m gpu8v6.zero_support --stage formal; python -m gpu8v6.support_line --stage test",
        "python -m gpu8v6.report_z; python -m gpu8v6.report_s; python -m gpu8v6.run_permutations; python -m gpu8v6.source_report",
        "python -m gpu8v6.external6; python -m gpu8v6.external_learned; python -m gpu8v6.clinvar_chain; python -m gpu8v6.temporal6",
        "python -m gpu8v6.temporal_support; python -m gpu8v6.rewire_control; python -m gpu8v6.expression6; python -m gpu8v6.kdgene_curve",
        "python -m gpu8v6.gene_block_ablation; python -m gpu8v6.family_audit; python -m gpu8v6.figures6",
        "python -m gpu8v6.numbers6; python -m gpu8v6.manuscript6; python -m gpu8v6.build_docx6 --markdown <md>",
        "```", "",
        "## 主要结果", "", summary(v).split("## 按方案章节的执行结果")[0].split("## 一句话结论")[1].strip(), "",
        "## 需要作者决定或完成的事项", "",
        "- 160 对盲审需要两位审阅者实际完成；完成前稿件中的相应表述保持“已准备、待核查”。",
        "- 若要把外部证据写成确认性结果，需要登记一个本轮未参与设计的新外部来源后再评价。",
        "- 题目按 V6 方案暂定；来源组加权未获支持，题目与主要创新中未保留“证据来源感知”。",
        ""])


def main() -> int:
    v = json.loads((paths6.REPORTS / "manuscript_numbers_v6.json").read_text(encoding="utf-8"))
    paths6.REPORTS.mkdir(parents=True, exist_ok=True)
    (paths6.REPORTS / "V6_执行与结果报告.md").write_text(summary(v), encoding="utf-8")
    rows = closure_rows(v)
    tab = ["# V6 问题逐条关闭表（方案第 16 节）", "", "| 原问题 | 修复动作 | 关闭证据 | 状态 |", "|---|---|---|---|"]
    tab += ["| " + " | ".join(r) + " |" for r in rows]
    (paths6.REPORTS / "V6_关闭表.md").write_text("\n".join(tab) + "\n", encoding="utf-8")
    (paths6.REPORTS / "V6_未完成与受限项.md").write_text(not_done(), encoding="utf-8")
    (OUT_DIR / "交付说明_V6.md").write_text(delivery(v), encoding="utf-8")
    print("reports written:", [p.name for p in paths6.REPORTS.glob("V6_*.md")], "and", OUT_DIR / "交付说明_V6.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

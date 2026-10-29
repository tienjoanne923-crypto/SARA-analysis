"""V7.1 reports: status-and-issues document (md + docx), closure table, unfinished
items and the delivery note.  Numbers only from manuscript_numbers_v71.json."""
from __future__ import annotations

import json
from pathlib import Path

from gpu8 import paths as p5

from . import paths71
from .appendix71 import PLAN_ROWS, UNFINISHED
from .common71 import dump_json, now_utc
from .manuscript71 import N, OUT_DIR
from .manuscript71_results import g, gi, gp, ci


def status_doc(n: N) -> str:
    L = []
    a = L.append
    a("# V7.1 现状说明与问题清单（第七轮多模态实测之后）")
    a(f"\n生成时间：{now_utc()}。所有数字由程序从 `gpu8/runs_v71/*/result.json` 与 `gpu8/results_v71/` 读取，逐项来源见 `reports_v71/manuscript_numbers_v71.csv`。\n")
    a("## 一、一页总结\n")
    a("### 1.1 这一轮做了什么\n")
    a(f"- 取得并登记真实多模态资源：HPA 共识组织表达、UniProt 参考蛋白组、MANE v1.5、IMPC 表型调用、MP 本体、MGI 同源、MGI 人工整理 MP–HP 映射、HPA 图像特征与亚细胞定位；冻结 SapBERT 与 ESM-2 35M 的固定提交版本。")
    a(f"- 建成按主键对齐的多模态特征包：序列 {gi(n, 'cov_gene_sequence_n')} 个基因、功能 {gi(n, 'cov_gene_function_n')}、表达 {gi(n, 'cov_gene_expression_n')}、图像 {gi(n, 'cov_gene_image_n')}；疾病文本 100%、HPO {gp(n, 'cov_dis_d_hpo')}、组织条件 {gp(n, 'cov_dis_d_tissue')}（有标注疾病）。")
    a(f"- 实现有类型的多模态核心（语义投影共享、序列独立适配、组织表达匹配、仅读可用性的门控、valid 位），以 {gi(n, 'fun_anchor_terms')} 个 GO 功能锚对齐序列，再以多正例排序学习检索；第 0 折开发选定 r = {gi(n, 'dev_sel_rank')}、λ_A = {g(n, 'dev_sel_lambda', 1)}、全局权重，三类模态经开发集配对置换全部保留。")
    a(f"- 在任何测试评价之前登记 E1–E8、强对照、准入规则与新确认集（{gi(n, 'conf_diseases')} 个疾病、{gi(n, 'conf_pairs')} 对）；完成 {gi(n, 'ledger_main_chain')} 次主核心整链拟合与全部变体、二级模型、行为检验与扩展通道。")
    a("\n### 1.2 效果：准确率直观数字（五折 4,238 个测试疾病）\n")
    a("| 方法 | Recall@10 | Hit@10 | Recall@50 | 最终 500 覆盖 |")
    a("|---|---|---|---|---|")
    for s, lab in (("f7", "F7（V6 固定融合）"), ("v6lgb", "V6 LightGBM（V6 主模型）"), ("chlgb", "仅检索通道 LightGBM（强对照）"),
                   ("srff", "同资源零训练固定融合"), ("core", "多模态核心单独"), ("lffix", "后融合（固定）"), ("lflgb", "后融合（LightGBM）")):
        a(f"| {lab} | {g(n, f'e1_{s}_recall10_end')} | {g(n, f'e1_{s}_hit10')} | {g(n, f'e1_{s}_recall50_end')} | {g(n, f'e1_{s}_coverage_final500')} |")
    a(f"\n新确认集（过滤评价）Recall@10：后融合 {g(n, 'e7_lflgb_recall10')}，V6 LightGBM {g(n, 'e7_v6lgb_recall10')}，仅通道 LightGBM {g(n, 'e7_chlgb_recall10')}，F7 {g(n, 'e7_f7_recall10')}。\n")
    a("### 1.3 三句话结论\n")
    a(f"1. 多模态核心能独立检索到 V6 参考路径看不到的候选：它自己的 500 名单覆盖 {g(n, 'e1_core_coverage_final500')}（V6 名单 {g(n, 'e1_v6lgb_coverage_final500')}），把 V6 结构上不可达的“原无路径”答案带进名单 {gp(n, 'e3_core_unseen_in500')}，单独 Recall@10 {g(n, 'e1_core_recall10_end')}（与 V6 主模型相当）。")
    a(f"2. 与 V6 袋外后融合后 Recall@10 为 {g(n, 'e1_lflgb_recall10_end')}，相对仅通道 LightGBM {ci(n, 'c1_lflgb_vs_chlgb_recall10_end')}，相对 V6 主模型 {ci(n, 'c1_lflgb_vs_v6lgb_recall10_end')}；预先冻结的新确认集上方向一致（{g(n, 'e7_lflgb_recall10')} 对 {g(n, 'e7_chlgb_recall10')}）。")
    a(f"3. 增益主体来自疾病语义与 GO 功能语义的对齐和后融合方法：功能置换使后融合下降 {g(n, 'e5_paired_permutation_F_lf_drop')}，组织表达 {g(n, 'e5_paired_permutation_E_lf_drop')}，序列 {g(n, 'e5_paired_permutation_S_lf_drop')}（区间跨零）；资源 × 方法 2×2 中新资源（ESM、HPA）只改变 {n.sd('e4_resource_gain_newmethod') if n.has('e4_resource_gain_newmethod') else '—'}；功能锚在测试上无可检测增益；图像与小鼠通道为阴性。")
    a("\n## 二、当前完整结果\n")
    a("详见稿件第 4.8–4.18 节与 `results_v71/tables/`：E1 `e1_main.csv`、`e1_contrasts.csv`；E2 `e2_same_data.csv`；E3 `e3_visibility.csv`；E4 `e4_resource_method.csv`；E5 `e5_behaviour.csv`；E6 `e6_attribution.csv`、`e6_hit_flow.json`；E7 `e7_confirmation.csv`、`e7_small_support.csv`；E8 `e8_optional.csv`。\n")
    a("### 2.1 模态行为（E5，五折平均，Recall@10 下降）\n")
    a("| 模态 | 关闭：核心 | 关闭：后融合 | 配对置换：核心 | 配对置换：后融合 | 只保留该模态：核心 R@10 |")
    a("|---|---|---|---|---|---|")
    for m, lab in (("F", "功能"), ("S", "序列"), ("E", "组织表达")):
        a(f"| {lab} | {g(n, f'e5_shutdown_{m}_core_drop')} | {g(n, f'e5_shutdown_{m}_lf_drop')} | {g(n, f'e5_paired_permutation_{m}_core_drop')} | {g(n, f'e5_paired_permutation_{m}_lf_drop')} | {g(n, f'e5_only_{m}_core_recall10')} |")
    a("\n### 2.2 冷基因（E3，前十恢复）\n")
    a("| 答案可见性 | 阳性占比 | V6 LightGBM | 核心单独 | 后融合 |")
    a("|---|---|---|---|---|")
    for c, lab in (("visible_ctd", "参考可见"), ("visible_hpo_only", "仅基因 HPO"), ("ppi_only", "仅 PPI"), ("unseen", "无路径")):
        a(f"| {lab} | {gp(n, f'e3_v6lgb_{c}_share_of_positives')} | {gp(n, f'e3_v6lgb_{c}_top10')} | {gp(n, f'e3_core_{c}_top10')} | {gp(n, f'e3_lflgb_{c}_top10')} |")
    a("\n## 三、问题清单（现象 → 证据 → 判断 → 方向）\n")
    issues = [
        ("A1 功能语义是主要路径", f"功能置换下降 {g(n, 'e5_paired_permutation_F_core_drop')}，远大于序列与表达", "核心的检索能力大部分来自同一编码器下“疾病语义—GO 语义”的对应；GO 与疾病文献可能重叠", "下一轮做严格历史 GO 版本与仅实验证据 GO 的敏感性分析"),
        ("A2 序列贡献小", f"序列置换使核心下降 {g(n, 'e5_paired_permutation_S_core_drop')}，后融合下降 {g(n, 'e5_paired_permutation_S_lf_drop')}（区间跨零）；只保留序列时核心 R@10 {g(n, 'e5_only_S_core_recall10')}", "序列只经功能锚间接进入语义空间，对无 GO 的冷基因帮助有限", "尝试更大的 ESM、结构感知表示或序列—表型直接监督（需新的独立验证）"),
        ("A4 新资源贡献近零", f"E4：同一新方法下旧资源 {g(n, 'e4t_e4_on_recall10_end', 4)}、新资源 {g(n, 'e4t_e4_nn_recall10_end', 4)}", "本轮收益来自方法（语义对齐 + 独立检索 + 后融合），不是来自 ESM 与 HPA", "论文中不宣称“新原始模态带来增益”；下一轮若要证明序列/表达的价值，需要专门的冷基因任务"),
        ("A5 功能锚未在测试上显效", f"E2：完整核心相对 λ_A = 0 为 {ci(n, 'c2_e2_core_vs_e2_alt')}，后融合中为 {ci(n, 'c2_e2_lf_vs_e2_lfalt')}", "开发集上的锚增益没有外推到测试", "锚只作为开发选择记录，不作为有效机制"),
        ("A6 全背景对照更好", f"敏感性：全背景未标注对照相对默认 {ci(n, 'c8_e8_bg_vs_e8_core')}", "默认对照偏保守，但这是测试后才看到的结果", "下一轮在开发集上预先比较后再定"),
        ("A3 组织条件依赖别名表", f"{gi(n, 'tis_only_broad')} 个疾病只命中宽泛规则", "别名表是保守启发式，不是临床组织定位", "引入经验证的 HPO—解剖映射与否定解析"),
        ("B1 冷基因仍难进入前十", f"无路径层阳性前十恢复：核心单独 {gp(n, 'e3_core_unseen_top10')}，后融合 {gp(n, 'e3_lflgb_unseen_top10')}", "覆盖提升大于前十提升，二级模型仍偏向参考可见基因", "为冷基因单独校准或做分层后融合"),
        ("C1 后融合依赖工作池与名单内重排", f"V6 500 固定时加入核心特征即提高 {n.sd('e6_inlist_gain') if n.has('e6_inlist_gain') else '—'}；同池仅通道模型只有 {g(n, 'e6t_e6_poolch_recall10_end')}", "增益大部分来自核心分数对名单内部的重排，其余来自候选扩展", "保留三种归因比较；限制池大小的敏感性留待下一轮"),
        ("C2 被挤出的旧命中", f"相对 V6 LightGBM 每种子新增 {g(n, 'flow_lf_vs_v6_lightgbm_new_hits_total_per_seed', 0)}、挤出 {g(n, 'flow_lf_vs_v6_lightgbm_lost_hits_total_per_seed', 0)}", "约一半的新增被旧命中流失抵消", "研究挤出的病种与基因类别，考虑保护高置信旧命中"),
        ("D1 五折已被多轮使用", "五折数字只是内部估计", "新确认集是唯一未触碰集合，但样本较小（618 个疾病）", "继续积累时间窗口外部集并完成盲审"),
        ("D2 扩展通道证据弱", f"小鼠通道开发 AUC {g(n, 'mouse_dev_auc_known_vs_other')}，图像变体见 E8", "只作探索性结果", "人工抽查映射后再决定是否接入"),
        ("E1 工程", "附包脚本不在本机，按数据合同重新实现；HCOP 404", "实现与方案逐节对照（关闭表）", "保持账本与哈希校验"),
    ]
    for t, e, j, d in issues:
        a(f"**{t}**　现象与证据：{e}。判断：{j}。方向：{d}。\n")
    a("## 四、目前能说什么、不能说什么\n")
    a("- 能说：真实多模态资源已接入并通过数据与计算层验收；核心能在完整背景上独立检索；在同一评价口径下，后融合相对强对照的差与区间如表所示；配对置换下功能、序列与组织模态各自的作用大小。")
    a("- 不能说：多模态核心证明了致病机制；门控权重大小代表因果重要性；冻结编码器从未见过相关知识；五折结果等同于独立验证；小鼠与图像通道已被证实有效。")
    a("\n## 五、建议的下一轮（按优先级）\n")
    for i, x in enumerate(["扩大并时间隔离新确认集，完成人工盲审；", "GO 严格历史版本与仅实验证据敏感性；",
                           "HPO—解剖映射替代别名表；", "冷基因分层后融合与校准；", "小鼠映射人工抽查后重评 E8；",
                           "ProtST / BioTranslator 的可行复现。"]):
        a(f"{i + 1}. {x}")
    a("\n## 附录 A　文件位置\n")
    a("- 代码 `gpu8/src/gpu8v71/`；测试 `gpu8/tests/test_v71_*.py`；运行 `gpu8/runs_v71/`；结果 `gpu8/results_v71/`；登记与清单 `gpu8/manifests_v71/`；缓存 `gpu8/caches_v71/`；表 `gpu8/tables_v71/`；外部资源 `gpu8/external/v71/`；稿件 `manuscript_v71/`。")
    return "\n".join(L)


def closure_doc() -> str:
    L = ["# V7.1 关闭表（方案逐节执行对照）", "", "| 方案节 | 要求 | 落实 | 状态 |", "|---|---|---|---|"]
    L += [f"| {a} | {b} | {c} | {d} |" for a, b, c, d in PLAN_ROWS]
    return "\n".join(L)


def unfinished_doc() -> str:
    return "# V7.1 未完成与受限项\n\n" + "\n".join(f"{i + 1}. {x}" for i, x in enumerate(UNFINISHED))


def delivery_note(n: N, ms: dict) -> str:
    return f"""# 交付说明（第七轮 V7.1）

生成时间：{now_utc()}

## 文件

- 稿件：`{Path(ms.get('markdown', '')).name}` 与同名 `.docx`（公式为 Word 原生 OMML 对象 {ms.get('omml')} 个；表 {ms.get('tables')} 张；图 {ms.get('images')} 幅），位于 `manuscript_v71/`，图在 `manuscript_v71/figures/`。
- 代码：`gpu8/src/gpu8v71/`（与 V5 `gpu8/src/gpu8/`、V6 `gpu8/src/gpu8v6/` 分开，二者未改动）；测试 `gpu8/tests/test_v71_core.py`、`test_v71_prep.py`。
- 结果：`gpu8/results_v71/`（audit、core、fusion、behavior、external、tables、figures）；逐运行权威结果 `gpu8/runs_v71/*/result.json`（{gi(n, 'ledger_runs')} 个，哈希通过 {gi(n, 'ledger_hash_ok')} 个）。
- 登记与清单：`gpu8/manifests_v71/`（models_v71、resources_v71、featurepack_v71、core_dev_selection_v71、protocol_v71_registered、fixed_late_beta_core、srff_v71、mouse_channel_v71、small_support_gamma_v71、run_ledger_v71）。
- 报告：`gpu8/reports_v71/`（V7.1 现状说明与问题清单 md/docx、关闭表、未完成与受限项、manuscript_numbers_v71.csv）。
- 新资源：`gpu8/external/v71/`（逐项 SHA-256 见 download_log_v71.csv）；编码器依赖安装在 `gpu8/v71_libs/`，冻结的 `.venv` 未改动。

## 复现顺序（Windows PowerShell）

```
$env:PYTHONPATH='D:\\信\\gpu8\\src'
powershell -File gpu8\\tools\\download_external_v71.ps1; powershell -File gpu8\\tools\\download_external_v71.ps1 -Large
python -m gpu8v71.resources71 --stage models; python -m gpu8v71.sequences71; python -m gpu8v71.encoders71
python -m gpu8v71.function71; python -m gpu8v71.expression71; python -m gpu8v71.image71; python -m gpu8v71.featurepack71
python -m gpu8v71.probe71; python -m gpu8v71.mouse71; python -m gpu8v71.chain71 --stage dev
python -m gpu8v71.resources71 --stage log; python -m gpu8v71.register71; python -m gpu8v71.baseline71 --stage tune
python -m gpu8v71.chain71 --stage chain --variants core --seeds 42 52 62
python -m gpu8v71.chain71 --stage chain --variants core_altanchor two_tower core_oldres core_strict core_bgctrl core_image core_imgloc core_imgcentred core_mouse --seeds 42
powershell -File gpu8\\tools\\run_v71_after_chain.ps1
python -m gpu8v71.reports71
```
"""


def main() -> int:
    v = json.loads((paths71.REPORTS / "manuscript_numbers_v71.json").read_text(encoding="utf-8"))["values"]
    n = N(v)
    R = paths71.REPORTS
    (R / "V71_现状说明与问题清单.md").write_text(status_doc(n), encoding="utf-8")
    (R / "V71_关闭表.md").write_text(closure_doc(), encoding="utf-8")
    (R / "V71_未完成与受限项.md").write_text(unfinished_doc(), encoding="utf-8")
    from gpu8v6.build_docx6 import build as build_docx
    st = build_docx(R / "V71_现状说明与问题清单.md", R / "V71_现状说明与问题清单.docx", paths71.FIGURES)
    ms = json.loads((R / "docx_build_report_v71.json").read_text(encoding="utf-8")) if \
        (R / "docx_build_report_v71.json").exists() else {}
    md = next(iter(OUT_DIR.glob("*_V71_*.md")), None)
    ms_info = {"markdown": str(md) if md else "", "omml": ms.get("verified_omml_objects"),
               "tables": ms.get("verified_tables"), "images": ms.get("images")}
    (OUT_DIR / "交付说明_V71.md").write_text(delivery_note(n, ms_info), encoding="utf-8")
    print(json.dumps({"status_docx": st.get("path"), "tables": st.get("verified_tables")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

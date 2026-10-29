"""Results 4.8-4.18, discussion, limitations and conclusion of the V7.1 manuscript.

Wording that states a direction is tied to the confidence interval through
``verdict``; interpretive sentences were written after the numbers were known and
say exactly what the tables show (including the negative and null results)."""
from __future__ import annotations

from .manuscript71_body import M_NAME, row, verdict


def g(n, k, d=3, default="—"):
    return n.f(k, d) if n.has(k) else default


def gp(n, k, d=1, default="—"):
    return n.pc(k, d) if n.has(k) else default


def gi(n, k, default="—"):
    return n.i(k) if n.has(k) else default


def ci(n, pre, d=3):
    return n.ci(pre, d) if n.has(pre + "_ci_low") else "（未得到）"


def vd(n, pre):
    return verdict(n, pre) if n.has(pre + "_ci_low") else ""


def res_resources(n) -> str:
    t = ["### 4.8 多模态资源与覆盖",
         f"表 10 给出各模态在项目主键上的实际覆盖（图 10A、B）。代表蛋白中 {gi(n, 'seq_reason_mane_select')} 个来自 MANE Select、{gi(n, 'seq_reason_uniprot_swissprot_canonical')} 个来自唯一的 UniProt Swiss-Prot canonical；{gi(n, 'seq_n_exceptions')} 个登录号对应多个基因而进入异常表，相应基因置缺失。{gi(n, 'seq_n_protein_coding_background')} 个蛋白编码基因中 {gi(n, 'seq_n_protein_coding_with_sequence')} 个有序列；{gi(n, 'seq_n_longer_than_1022')} 个蛋白长于 1,022 个残基，按式 (16) 分窗。ESM-2 35M 对全部序列的一次前向用时 {g(n, 'enc_esm_seconds', 0)} 秒。SapBERT 编码的 {gi(n, 'enc_disease_n')} 个疾病模板中有 {gi(n, 'enc_disease_trunc')} 个超过 128 个 token 被截断（名称与定义在前，截断的是尾部同义词），HPO 与 GO 词条的截断数分别为 {gi(n, 'enc_hpo_trunc')} 与 {gi(n, 'enc_go_trunc')}。",
         f"功能锚共 {gi(n, 'fun_anchor_terms')} 个 GO 词条，覆盖 {gi(n, 'fun_anchor_genes')} 个基因（每个基因的中位锚数 {g(n, 'fun_anchor_median', 0)}）。HPA 共识表达覆盖 {gi(n, 'expr_genes')} 个基因，其中 {gi(n, 'expr_zero_cells')} 个“实测为零”的单元作为观测保留。登记的 {gi(n, 'tis_n_rules')} 条别名规则为 {gi(n, 'tis_diseases')} 个疾病给出组织条件（有标注疾病 {gi(n, 'tis_labelled')}/{gi(n, 'tis_labelled_n')}），其中 {gi(n, 'tis_only_broad')} 个只命中宽泛规则。{gi(n, 'cov_gene_none')} 个基因三类基因侧模态全缺，始终留在评价背景中并以明示回退处理。",
         "| 模态 | 编码与来源 | 覆盖数 | 占全部基因 / 有标注疾病 | 占已知阳性基因 |",
         "|---|---|---|---|---|",
         f"| 蛋白序列 | ESM-2 35M，MANE / UniProt | {gi(n, 'cov_gene_sequence_n')} | {gp(n, 'cov_gene_sequence')} | {gp(n, 'cov_gene_sequence_pos')} |",
         f"| 功能语义 | SapBERT × GO 直接注释 | {gi(n, 'cov_gene_function_n')} | {gp(n, 'cov_gene_function')} | {gp(n, 'cov_gene_function_pos')} |",
         f"| 组织表达 | HPA 共识 51 组织 | {gi(n, 'cov_gene_expression_n')} | {gp(n, 'cov_gene_expression')} | {gp(n, 'cov_gene_expression_pos')} |",
         f"| 图像（扩展） | HPA DenseNet 1,024 维 | {gi(n, 'cov_gene_image_n')} | {gp(n, 'cov_gene_image')} | {gp(n, 'cov_gene_image_pos')} |",
         f"| 小鼠表型（扩展） | IMPC + MGI 1:1 + MP–HP | {gi(n, 'mouse_genes_with_model')} | — | — |",
         f"| 疾病文本 | SapBERT 掩蔽模板 | — | {gp(n, 'cov_dis_d_text')} | — |",
         f"| 疾病 HPO | SapBERT 频率加权 | — | {gp(n, 'cov_dis_d_hpo')} | — |",
         f"| 疾病组织条件 | 别名表 q_d | — | {gp(n, 'cov_dis_d_tissue')} | — |",
         "",
         "表 10　多模态特征包覆盖（数据层验收）。所有数组已通过维数、掩码置零、有限值与非恒定分布检查；图像与小鼠为有条件扩展。",
         f"本机 GPU 探针（全部 {gi(n, 'seq_n_genes_background')} 个基因 × 16 个真实疾病、开启图像分支、{gi(n, 'probe_params_img')} 个可训练参数）每次前向加反向更新约 {g(n, 'probe_update_s', 3)} 秒，GPU 峰值分配 {g(n, 'probe_gpu', 0)} MiB，进程峰值常驻内存 {g(n, 'probe_rss', 0)} MiB。"]
    return "\n\n".join(t)


def res_dev(n) -> str:
    names = [("r64_lam0.0_global", "64 / 0 / 全局"), ("r64_lam0.0_small", "64 / 0 / 门控"),
             ("r64_lam0.2_global", "64 / 0.2 / 全局"), ("r64_lam0.2_small", "64 / 0.2 / 门控"),
             ("r128_lam0.0_global", "128 / 0 / 全局"), ("r128_lam0.0_small", "128 / 0 / 门控"),
             ("r128_lam0.2_global", "128 / 0.2 / 全局"), ("r128_lam0.2_small", "128 / 0.2 / 门控")]
    grid = "；".join(f"{lab} {g(n, 'dev_' + k)}" for k, lab in names if n.has("dev_" + k))
    sel = f"r = {gi(n, 'dev_sel_rank')}、λ_A = {g(n, 'dev_sel_lambda', 1)}、全局权重"
    t = ["### 4.9 开发选择、模态保留与登记",
         f"第 0 折开发集（864 个开发家族疾病，全背景）上 8 个预定配置的 Recall@10 为：{grid}（维数 / 锚权重 / 门控；图 10C）。最佳配置为 {sel}，其余配置都没有落在它的一个配对标准误之内，因此一标准误差规则也选择它；开发集上锚对齐在两个维数下都提高了指标，小门控反而低于全局权重。所选核心的实际可训练参数为 {gi(n, 'core_params_selected')} 个（$r=128$、全局权重；$r=64$ 的规格值 80,069 与带图像的 145,670 由单元测试核对）。",
         f"开发集配对置换决定保留哪些模态：置换功能向量使 Recall@10 下降 {g(n, 'dev_perm_F')}（配对标准误 {g(n, 'dev_perm_F_se')}），置换序列下降 {g(n, 'dev_perm_S')}（{g(n, 'dev_perm_S_se')}），在疾病之间交换组织权重下降 {g(n, 'dev_perm_E')}（{g(n, 'dev_perm_E_se')}）；三者都超过一个标准误，全部保留。单模态关闭的下降分别为 {g(n, 'dev_off_F')}、{g(n, 'dev_off_S')} 与 {g(n, 'dev_off_E')}。准入检查中，开发集 {gi(n, 'adm_dev_positives')} 个阳性里 V6 500 名单含 {gi(n, 'adm_in_v6_500')} 个，核心前 500 另外带来 {gi(n, 'adm_added_by_core_top500')} 个（覆盖增加 {gp(n, 'adm_union_coverage_gain')}），核心进入后融合。",
         f"上述选择、特征包与编码器哈希、资源日志以及新确认集哈希于 {n.raw('registered_utc') if n.has('registered_utc') else '—'} 登记；账本中第一次 V7.1 测试评价开始于 {n.raw('ledger_first_test_utc') if n.has('ledger_first_test_utc') else '—'}，晚于登记时间。同资源固定融合的非负权重（功能 {g(n, 'srff_w_F', 1)}、表达 {g(n, 'srff_w_E', 1)}、序列近邻转移 {g(n, 'srff_w_S', 1)}）与固定后融合的 β = {g(n, 'lf_beta', 2)} 也只在开发集上确定。"]
    return "\n\n".join(t)


def res_e1(n) -> str:
    cols = ["recall10_end", "hit10", "recall50_end", "ndcg10_end", "coverage_final500", "oracle_recall10"]
    rows = ["| 方法 | Recall@10 | Hit@10 | Recall@50 | NDCG@10 | 最终 500 覆盖 | U@10 |", "|---|---|---|---|---|---|---|"]
    for s in ("f7", "v6lgb", "chlgb", "srff", "core", "lffix", "lflgb"):
        rows.append(row(n, s, cols))
    t = ["### 4.10 E1：外层冷启动主比较",
         "表 11 在同一 4,238 个五折测试疾病上比较七个方法（图 11）；学习型方法先在每个种子上逐疾病计算，再对三个种子平均。V6 的方法在其 500 名单上排序，核心单独与同资源固定融合使用各自的 500 名单，后融合在工作池（V6 500 ∪ 核心前 500）上一次排序，“最终 500 覆盖”统一取最终排序的前 500；U@10 为各自候选内的 oracle 上界。",
         "\n".join(rows), "",
         "表 11　E1 主比较（五折 × 三种子；确定性方法单次）。",
         f"**核心单独。** 多模态核心独立检索的 Recall@10 为 {g(n, 'e1_core_recall10_end')}，与 V6 主模型（{g(n, 'e1_v6lgb_recall10_end')}）相当，但它的 500 名单覆盖 {g(n, 'e1_core_coverage_final500')} 高于 V6 名单的 {g(n, 'e1_v6lgb_coverage_final500')}，Recall@50（{g(n, 'e1_core_recall50_end')}）也略高；它的 Hit@10（{g(n, 'e1_core_hit10')}）低于 V6 LightGBM（{g(n, 'e1_v6lgb_hit10')}），说明核心更擅长把答案带进名单，而不是把它们排到最前。核心相对同资源零训练固定融合的差为 {ci(n, 'c1_core_vs_srff_recall10_end')}，{vd(n, 'c1_core_vs_srff_recall10_end')}：同样的输入不经学习对齐只能达到 {g(n, 'e1_srff_recall10_end')}，其 500 覆盖只有 {g(n, 'e1_srff_coverage_final500')}。",
         f"**后融合。** LightGBM 后融合的 Recall@10 为 {g(n, 'e1_lflgb_recall10_end')}：相对重新登记的仅检索通道 LightGBM 为 {ci(n, 'c1_lflgb_vs_chlgb_recall10_end')}（{vd(n, 'c1_lflgb_vs_chlgb_recall10_end')}），相对 F7 为 {ci(n, 'c1_lflgb_vs_f7_recall10_end')}，相对存档 V6 LightGBM 为 {ci(n, 'c1_lflgb_vs_v6lgb_recall10_end')}。同一对比下 Hit@10 的差为 {ci(n, 'c1_lflgb_vs_chlgb_hit10')}，Recall@50 为 {ci(n, 'c1_lflgb_vs_chlgb_recall50_end')}，最终 500 覆盖为 {ci(n, 'c1_lflgb_vs_chlgb_coverage_final500')}。逐病看，{gp(n, 'c1_lflgb_vs_chlgb_recall10_end_share_a_better')} 的疾病后融合更好、{gp(n, 'c1_lflgb_vs_chlgb_recall10_end_share_b_better')} 更差，其余持平。固定后融合（β = {g(n, 'lf_beta', 2)}）相对 F7 为 {ci(n, 'c1_lffix_vs_f7_recall10_end')}，LightGBM 后融合又比固定后融合高 {ci(n, 'c1_lflgb_vs_lffix_recall10_end')}。重新登记的仅检索通道 LightGBM 相对 V6 主 LightGBM（含 GO/通路/表达 SVD 特征）为 {ci(n, 'c1_chlgb_vs_v6lgb_recall10_end')}，与 V6 的事后分析一致：旧的原始功能与表达拼接并没有帮助 V6 主模型。"]
    return "\n\n".join(t)


def res_e2(n) -> str:
    items = [("e2_core", "核心（完整对齐）单独"), ("e2_alt", "核心（锚权重翻转，λ_A = 0）单独"), ("e2_tt", "线性双塔单独"),
             ("e2_srff", "同资源固定融合"), ("e2_lf", "后融合 LGB（完整核心）"), ("e2_lfalt", "后融合 LGB（锚翻转）"),
             ("e2_lftt", "后融合 LGB（双塔）"), ("e2_raw", "原始拼接 LGB（同一工作池）")]
    rows = ["| 方法（种子 42） | Recall@10 | Hit@10 | 最终 500 覆盖 |", "|---|---|---|---|"]
    for s, lab in items:
        rows.append(f"| {lab} | {g(n, f'e2t_{s}_recall10_end')} | {g(n, f'e2t_{s}_hit10')} | {g(n, f'e2t_{s}_coverage_final500')} |")
    t = ["### 4.11 E2：同数据、不同算法",
         "表 12 固定全部输入，只改变算法（图 14B）。锚权重翻转把开发选定的 λ_A = 0.2 换成 0（去掉锚对齐）；线性双塔把疾病侧 [文本, HPO, 组织] 与基因侧 [序列, 功能, 表达] 各自线性投影后做内积，没有类型、门控与锚；原始拼接把 PCA 压缩的模态原始向量与疾病向量直接交给 LightGBM，在同一工作池上排序。",
         "\n".join(rows), "", "表 12　E2 同数据不同算法（种子 42，五折）。",
         f"有类型核心相对线性双塔为 {ci(n, 'c2_e2_core_vs_e2_tt')}，相对同资源固定融合为 {ci(n, 'c2_e2_core_vs_e2_srff')}，两者的{vd(n, 'c2_e2_core_vs_e2_tt')}：按模态分设适配与匹配分数确实优于把所有向量线性压在一起。原始拼接在同一工作池上只有 {g(n, 'e2t_e2_raw_recall10_end')}，后融合完整核心相对它为 {ci(n, 'c2_e2_lf_vs_e2_raw')}，与 V6“原始拼接没有帮助”的结论一致——增益来自对齐后的匹配分数，而不是把原始向量交给树模型。",
         f"**锚对齐在测试上未显示增益。** 开发集上锚对齐提高了指标，但在五折测试上，完整核心相对 λ_A = 0 的差为 {ci(n, 'c2_e2_core_vs_e2_alt')}，进入后融合后为 {ci(n, 'c2_e2_lf_vs_e2_lfalt')}，{vd(n, 'c2_e2_lf_vs_e2_lfalt')}。按登记的降级规则，功能锚只能被描述为开发集上的选择，不能写成已证实的外层收益。"]
    return "\n\n".join(t)


def res_e3(n) -> str:
    cls = [("visible_ctd", "参考关联可见"), ("visible_hpo_only", "仅基因 HPO"), ("ppi_only", "仅 PPI 邻居"),
           ("unseen", "原无路径")]
    rows = ["| 答案可见性 | 阳性占比 | V6 LGB 入500 | 核心入500 | 后融合入500 | V6 LGB 前10 | 核心前10 | 后融合前10 | 严格冷核心前10 | 严格冷后融合前10 |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for c, lab in cls:
        rows.append(f"| {lab} | {gp(n, f'e3_v6lgb_{c}_share_of_positives')} | {gp(n, f'e3_v6lgb_{c}_in500')} | "
                    f"{gp(n, f'e3_core_{c}_in500')} | {gp(n, f'e3_lflgb_{c}_in500')} | {gp(n, f'e3_v6lgb_{c}_top10')} | "
                    f"{gp(n, f'e3_core_{c}_top10')} | {gp(n, f'e3_lflgb_{c}_top10')} | {gp(n, f'e3_strictcore_{c}_top10')} | "
                    f"{gp(n, f'e3_strictlf_{c}_top10')} |")
    t = ["### 4.12 E3：关联冷基因与更严格的基因监督冷",
         "沿用 V6 事后定义的答案可见性分层（只作诊断，不作路由输入；PPI 为 V6 来源引擎使用的物理互作网络），逐个测试阳性报告它是否进入最终 500 与前十（表 13；图 12A）。“严格冷”变体把锚训练限制在当前上下文关联可见的基因上，关联冷基因只保留冻结特征。",
         "\n".join(rows), "", "表 13　E3 按答案可见性分层的恢复（学习型方法三种子平均；严格冷变体为种子 42）。",
         f"这里是多模态核心最清楚的作用：V6 的检索路径对“原无路径”的答案在结构上不可达（入 500 {gp(n, 'e3_v6lgb_unseen_in500')}），核心把其中 {gp(n, 'e3_core_unseen_in500')} 带进自己的 500 名单、{gp(n, 'e3_core_unseen_top10')} 排进前十；“仅 PPI 邻居”的答案入 500 由 {gp(n, 'e3_v6lgb_ppi_only_in500')} 提高到 {gp(n, 'e3_core_ppi_only_in500')}，前十由 {gp(n, 'e3_v6lgb_ppi_only_top10')} 提高到 {gp(n, 'e3_core_ppi_only_top10')}。后融合保留了大部分覆盖（入 500 分别为 {gp(n, 'e3_lflgb_unseen_in500')} 与 {gp(n, 'e3_lflgb_ppi_only_in500')}），但它的前十更偏向参考可见的基因（冷层前十 {gp(n, 'e3_lflgb_unseen_top10')} 与 {gp(n, 'e3_lflgb_ppi_only_top10')}，低于核心单独），同时把参考可见层的前十由 {gp(n, 'e3_v6lgb_visible_ctd_top10')} 提高到 {gp(n, 'e3_lflgb_visible_ctd_top10')}。严格冷变体在两层冷阳性上的前十恢复与默认设置接近（核心单独 {gp(n, 'e3_strictcore_unseen_top10')} / {gp(n, 'e3_strictcore_ppi_only_top10')}），说明冷基因的恢复不依赖它们自身的锚监督，而主要来自冻结的功能语义输入（E5）。"]
    return "\n\n".join(t)


def res_e4(n) -> str:
    t = ["### 4.13 E4：资源 × 方法",
         "“旧资源”指 V6 已有的 GO 功能与 GTEx 表达（新方法中以 SapBERT 编码 GO，GTEx 映射到同一 51 组织轴，不用 ESM 与 HPA）；“旧方法”指 LightGBM 直接吃特征。四格均为种子 42、五折（图 14C）：",
         "| | 旧方法 | 新方法 |", "|---|---|---|",
         f"| 旧资源 | V6 LightGBM：{g(n, 'e4t_e4_oo_recall10_end')} | 旧资源核心后融合：{g(n, 'e4t_e4_on_recall10_end')} |",
         f"| 新资源 | LightGBM + 原始模态向量：{g(n, 'e4t_e4_no_recall10_end')} | 完整核心后融合：{g(n, 'e4t_e4_nn_recall10_end')} |",
         "", "表 14　E4 资源 × 方法 2×2（Recall@10）。",
         f"方法的作用远大于资源：同一旧资源下换用新方法提高 {n.sd('e4_method_gain_oldres') if n.has('e4_method_gain_oldres') else '—'}，同一新资源下提高 {n.sd('e4_method_gain_newres') if n.has('e4_method_gain_newres') else '—'}；而同一方法下换用新资源（加入 ESM 序列、以 HPA 替换 GTEx）只改变 {n.sd('e4_resource_gain_newmethod') if n.has('e4_resource_gain_newmethod') else '—'}（新方法）与 {n.sd('e4_resource_gain_oldmethod') if n.has('e4_resource_gain_oldmethod') else '—'}（旧方法）。也就是说，本轮后融合的主要收益来自“用同一冻结语义编码器把疾病描述与 GO 功能对齐、再独立检索并袋外后融合”这一方法，而不是来自新增的蛋白序列与 HPA 表达资源。"]
    return "\n\n".join(t)


def res_e5(n) -> str:
    mods = [("F", "功能"), ("S", "序列"), ("E", "组织表达")]
    rows = ["| 模态 | 关闭：核心 ΔR@10 | 关闭：后融合 ΔR@10 | 配对置换：核心 ΔR@10 | 配对置换：后融合 ΔR@10 | 只保留该模态：核心 R@10 | 只保留该模态：后融合 R@10 |",
            "|---|---|---|---|---|---|---|"]

    def cell(m, p, w):
        k = f"e5_{p}_{m}_{w}_drop"
        if not n.has(k):
            return "—"
        lo, hi = n.v.get(k + "_ci_low"), n.v.get(k + "_ci_high")
        if lo is None:
            return n.f(k)
        n.used.update({k + "_ci_low", k + "_ci_high"})
        return f"{n.f(k)}（{lo:+.3f}, {hi:+.3f}）"
    for m, lab in mods:
        rows.append(f"| {lab} | {cell(m, 'shutdown', 'core')} | {cell(m, 'shutdown', 'lf')} | "
                    f"{cell(m, 'paired_permutation', 'core')} | {cell(m, 'paired_permutation', 'lf')} | "
                    f"{g(n, f'e5_only_{m}_core_recall10')} | {g(n, f'e5_only_{m}_lf_recall10')} |")
    t = ["### 4.14 E5：模型是否真的使用了模态",
         f"在固定模型与固定候选下（种子 42 的外层核心与后融合 LightGBM，五折），表 15 报告相对正常输入的 Recall@10 下降（正值 = 下降，括号内为家族 bootstrap 95% 区间；图 13）；正常输入下核心单独为 {g(n, 'e5_none_-_core_recall10')}，后融合为 {g(n, 'e5_none_-_lf_recall10')}。配对置换保持基因编号、标签与其他通道不变，只交换指定输入的对应关系，并保留长度、度或组织数等边际分布（三次抽样平均）。",
         "\n".join(rows), "", "表 15　E5 模态行为（五折平均）。",
         f"**功能语义是主要路径。** 置换功能向量使核心单独下降 {g(n, 'e5_paired_permutation_F_core_drop')}（几乎失去全部检索能力），使后融合下降 {g(n, 'e5_paired_permutation_F_lf_drop')}，降到仅通道模型之下——后融合已经学会依赖核心分数，错配的核心分数会误导它；只保留功能模态时，后融合几乎不变（{g(n, 'e5_only_F_lf_recall10')}）。",
         f"**序列与表达的作用小。** 在核心单独时，序列与组织权重的配对置换都使指标下降且区间在零以上（{g(n, 'e5_paired_permutation_S_core_drop')} 与 {g(n, 'e5_paired_permutation_E_core_drop')}），说明核心确实用到了这两种对应；但在后融合中，组织权重置换的下降为 {g(n, 'e5_paired_permutation_E_lf_drop')}（区间在零以上），序列置换的下降为 {g(n, 'e5_paired_permutation_S_lf_drop')}，区间跨零。只保留序列或只保留表达时，核心单独分别只剩 {g(n, 'e5_only_S_core_recall10')} 与 {g(n, 'e5_only_E_core_recall10')}。",
         f"**不是覆盖标志。** 在同一工作池上，只给二级模型核心可用位与组织覆盖、不给核心分数的“仅掩码”模型 Recall@10 为 {g(n, 'e6t_e6_mask_recall10_end')}，与同池仅通道模型（{g(n, 'e6t_e6_poolch_recall10_end')}）相当，远低于后融合（{g(n, 'e6t_e6_poollf_recall10_end')}），说明增益不是由“哪些基因有模态”这一研究热度信号带来的。全背景分块不变性在真实输入上的最大绝对差为 {n.raw('chunk_invariance_diff') if n.has('chunk_invariance_diff') else '—'}。",
         f"**泄漏敏感性。** 把疾病模板改用 V6 二级掩蔽（额外去掉多词基因全名）后，固定核心的 Recall@10 由 {g(n, 'sens_normal')} 变为 {g(n, 'sens_text_mask_level2')}；把功能向量换成 2023-10-09 版 GO 与 GOA（{gi(n, 'sens_go2023_genes')} 个基因有注释）后为 {g(n, 'sens_historical_go_2023')}。两者变化都很小：核心的检索能力不是来自文本中残留的基因名，也不依赖可能源自被评价关系的最新 GO 注释。"]
    return "\n\n".join(t)


def res_e6(n) -> str:
    items = [("e6_v6ch", "V6 500 固定：仅通道 LGB"), ("e6_v6core", "V6 500 固定：LGB + 核心特征"),
             ("e6_poolf7", "共享新池：F7"), ("e6_poolch", "共享新池：仅通道 LGB"),
             ("e6_poollf", "共享新池：后融合 LGB"), ("e6_mask", "共享新池：仅掩码 LGB")]
    rows = ["| 比较（种子 42） | Recall@10 | Hit@10 | 最终 500 覆盖 |", "|---|---|---|---|"]
    for s, lab in items:
        rows.append(f"| {lab} | {g(n, f'e6t_{s}_recall10_end')} | {g(n, f'e6t_{s}_hit10')} | {g(n, f'e6t_{s}_coverage_final500')} |")
    t = ["### 4.15 E6：候选归因与命中流量",
         "三种归因比较（表 16）分别回答：旧 500 固定、只换评分时核心特征的池内作用；共享新池时不同评分器的算法差异；各自端到端时的整体作用。",
         "\n".join(rows), "", "表 16　E6 候选归因。",
         f"后融合的收益不只是“候选更多”：在完全不改变 V6 500 名单的条件下，只把核心特征加进 LightGBM，Recall@10 就提高 {n.sd('e6_inlist_gain') if n.has('e6_inlist_gain') else '—'}；反过来，把工作池交给只用 V6 通道的模型（或 F7），指标几乎停在 V6 名单上的水平（{g(n, 'e6t_e6_poolch_recall10_end')} / {g(n, 'e6t_e6_poolf7_recall10_end')}），同池后融合比它高 {n.sd('e6_pool_gain') if n.has('e6_pool_gain') else '—'}。新加入工作池的基因几乎不带 V6 通道信号，只有核心分数能把其中的答案排上来。",
         f"与存档 V6 LightGBM 的逐种子前十相比，后融合每个种子平均新增 {g(n, 'flow_lf_vs_v6_lightgbm_new_hits_total_per_seed', 0)} 个命中、挤出 {g(n, 'flow_lf_vs_v6_lightgbm_lost_hits_total_per_seed', 0)} 个、保留 {g(n, 'flow_lf_vs_v6_lightgbm_kept_hits_total_per_seed', 0)} 个；{gp(n, 'flow_lf_vs_v6_lightgbm_diseases_gaining')} 的疾病净增命中，{gp(n, 'flow_lf_vs_v6_lightgbm_diseases_losing')} 的疾病净减（图 12B）。相对仅检索通道 LightGBM，新增 {g(n, 'flow_lf_vs_channel_lgb_new_hits_total_per_seed', 0)}、挤出 {g(n, 'flow_lf_vs_channel_lgb_lost_hits_total_per_seed', 0)}。被挤出的旧命中约为新增命中的一半，这是后融合应当继续改进的地方。"]
    return "\n\n".join(t)


def res_e7(n) -> str:
    rows = ["| 方法 | 新确认集 Recall@10 | Hit@10 | Recall@50 | 新标签在名单/池中的比例 |", "|---|---|---|---|---|"]
    for s in ("f7", "v6lgb", "chlgb", "srff", "core", "lffix", "lflgb"):
        rows.append(f"| {M_NAME[s]} | {g(n, f'e7_{s}_recall10')} | {g(n, f'e7_{s}_hit10')} | {g(n, f'e7_{s}_recall50')} | {g(n, f'e7_{s}_coverage_in_list')} |")
    gamma = n.v.get("ss_gamma")
    gk = f"joint_core_g{gamma:g}" if gamma is not None else "joint_core_g0.5"
    ss = [("sem", "零支持语义（V6 融合）"), ("rwr", "支持基因 RWR"), ("joint", "V6 联合"), ("core", "多模态核心"),
          (gk, "V6 联合 + 核心（γ 由开发集定）")]
    ssr = ["| 方法（k = 1） | Recall@10 | Recall@50 |", "|---|---|---|"]
    for k, lab in ss:
        ssr.append(f"| {lab} | {g(n, f'ss_{k}_recall10')} | {g(n, f'ss_{k}_recall50')} |")
    t = ["### 4.16 E7：支持任务与预先冻结的新确认集",
         f"新确认集为 {gi(n, 'conf_diseases')} 个五折疾病的 {gi(n, 'conf_pairs')} 个非 CTD 关系（GenCC {gi(n, 'conf_GenCC')} 对、ClinVar ≥2 星 {gi(n, 'conf_ClinVar')} 对），在建模前冻结；每个方法的已排名单删除该病 CTD 阳性后计算（表 17；图 14A）。它是本轮唯一未被任何设计或评价触碰过的标签集合。",
         "\n".join(rows), "", "表 17　E7 新确认集（过滤评价；学习型方法三种子平均）。",
         f"新确认集与五折结论方向一致：后融合相对仅检索通道 LightGBM 为 {ci(n, 'c7_lflgb_vs_chlgb')}，相对存档 V6 LightGBM 为 {ci(n, 'c7_lflgb_vs_v6lgb')}，相对 F7 为 {ci(n, 'c7_lflgb_vs_f7')}，区间都在零以上；核心单独相对同资源固定融合为 {ci(n, 'c7_core_vs_srff')}。由于样本只有 {gi(n, 'conf_diseases')} 个疾病，区间较宽。",
         f"V6 支持任务（≥6 个已知基因，k = 0/1/3/5）作为存档保留。V6 设计不能使用的 2–5 个已知基因疾病另做 k = 1 小支持集合（{gi(n, 'ss_joint_n_diseases')} 个疾病，每病 5 次抽样，排除支持基因后在全背景排序）：",
         "\n".join(ssr), "",
         f"表 18　E7 小支持集合（γ = {g(n, 'ss_gamma', 1)} 在开发集上选定，允许为 0）。",
         f"一个支持基因的 RWR 已明显超过零支持语义，V6 联合更高；再加入核心（γ = {g(n, 'ss_gamma', 1)}）使 Recall@10 由 {g(n, 'ss_joint_recall10')} 提高到 {g(n, f'ss_{gk}_recall10')}。开发集上的最优 γ 落在预定网格的上界 0.5，说明核心权重可能还可以更大；按协议本文不在测试后扩大网格。"]
    return "\n\n".join(t)


def res_e8(n) -> str:
    items = [("e8_core", "核心（种子 42）"), ("e8_img", "核心 + 图像"), ("e8_loc", "核心 + 仅定位标签（图像对照）"),
             ("e8_cent", "核心 + 细胞系中心化图像"), ("e8_mouse", "核心 + 小鼠表型"), ("e8_bg", "核心，全背景未标注对照（敏感性）")]
    rows = ["| 变体（单独检索） | Recall@10 | Hit@10 | 500 覆盖 | 相对核心 ΔR@10 |", "|---|---|---|---|---|"]
    for s, lab in items:
        d = ci(n, f"c8_{s}_vs_e8_core") if s != "e8_core" else "—"
        rows.append(f"| {lab} | {g(n, f'e8t_{s}_recall10_end')} | {g(n, f'e8t_{s}_hit10')} | {g(n, f'e8t_{s}_coverage_final500')} | {d} |")
    t = ["### 4.17 E8：图像与跨物种表型（探索性）",
         f"HPA 图像特征文件取得并解析：{gi(n, 'img_rows_read')} 张图像中去除多靶标或无法唯一归因的 {gi(n, 'img_multi_or_empty_gene_images')} 张后，{gi(n, 'img_genes_with_image')} 个基因在 {gi(n, 'img_cell_lines')} 个细胞系上有等权聚合的 1,024 维表征。小鼠通道由 {gi(n, 'mouse_impc_calls')} 条 IMPC 显著调用、{gi(n, 'mouse_one_to_one_classes')} 个 MGI 1:1 同源类与 {gi(n, 'mouse_bridge_rows')} 条 MGI 人工整理 MP–HP 映射构成；{gp(n, 'mouse_calls_mapped_share')} 的调用可经映射（{gp(n, 'mouse_calls_mapped_direct_share')} 直接映射）到 HPO，开发集已知对与其他对的 AUC 为 {g(n, 'mouse_dev_auc_known_vs_other')}，勉强超过预定的 0.55 门槛，因此只作探索性运行；人工抽查表已导出但未完成。",
         "\n".join(rows), "", "表 19　E8 可选通道（外层拟合，种子 42；单独检索，全背景）。",
         f"两条扩展都没有带来增益：加入图像后核心相对无图像版本为 {ci(n, 'c8_e8_img_vs_e8_core')}，仅定位标签对照为 {ci(n, 'c8_e8_loc_vs_e8_core')}，细胞系中心化图像为 {ci(n, 'c8_e8_cent_vs_e8_core')}；在相同优势细胞系内置换图像使图像变体下降 {g(n, 'e8p_core_image_drop')}。加入小鼠表型为 {ci(n, 'c8_e8_mouse_vs_e8_core')}，在调用广度相近的基因之间置换小鼠匹配的变化为 {g(n, 'e8p_core_mouse_drop')}，没有可检测作用。按登记规则两条扩展都不进入主模型，作为阴性结果保留。",
         f"作为预定的敏感性分析，把未标注对照从“参考关联中出现过的基因”换成全背景后，核心单独为 {ci(n, 'c8_e8_bg_vs_e8_core')}，{vd(n, 'c8_e8_bg_vs_e8_core')}。这提示默认的对照设置偏保守；由于它是测试后才看到的敏感性结果，本文不据此改变主模型，只作为下一轮的开发选项。"]
    return "\n\n".join(t)


def res_ledger(n) -> str:
    t = ["### 4.18 资源、账本与复现",
         f"V7.1 账本共 {gi(n, 'ledger_runs')} 个运行记录（完成 {gi(n, 'ledger_done')}、失败 {gi(n, 'ledger_failed')}），内容哈希全部通过的有 {gi(n, 'ledger_hash_ok')} 个；其中核心整链拟合 {gi(n, 'ledger_chain_fits')} 次（主核心 {gi(n, 'ledger_main_chain')} 次）。单次整链拟合平均 {g(n, 'chain_mean_seconds', 0)} 秒，GPU 峰值分配 {g(n, 'chain_peak_gpu', 0)} MiB，进程峰值常驻内存 {g(n, 'chain_peak_rss', 0)} MiB。V5 与 V6 的全部档案目录在本轮开始后没有任何文件被修改，冻结实验环境未安装任何新包（编码器依赖单独置于 v71_libs）。"]
    return "\n\n".join(t)


def results_v71(n) -> str:
    return "\n\n".join([res_resources(n), res_dev(n), res_e1(n), res_e2(n), res_e3(n), res_e4(n), res_e5(n),
                        res_e6(n), res_e7(n), res_e8(n), res_ledger(n)])


def discussion(n, claim, crit) -> str:
    t = ["## 5. 讨论",
         f"**独立检索的核心补上了 V6 看不到的候选。** V6 的剩余瓶颈是参考路径不可见的答案。核心不读取任何 V6 分数，只凭疾病语义、功能语义、蛋白序列与组织表达在两万基因背景上检索；它单独的 Recall@10 与 V6 主模型相当，但把 V6 结构上不可达的“原无路径”答案带进名单（{gp(n, 'e3_core_unseen_in500')}），并与 V6 名单高度互补——二者的工作池把最终 500 覆盖由 {g(n, 'e1_chlgb_coverage_final500')} 提高到 {g(n, 'e1_lflgb_coverage_final500')}。E6 进一步说明，后融合的增益一部分来自新候选，更大一部分来自核心分数对 V6 名单内部的重新排序（V6 500 固定时就提高 {n.sd('e6_inlist_gain') if n.has('e6_inlist_gain') else '—'}）。",
         f"**增益的主体是功能语义对齐，而不是新的原始模态。** 配对置换（功能 {g(n, 'e5_paired_permutation_F_lf_drop')}，表达 {g(n, 'e5_paired_permutation_E_lf_drop')}，序列 {g(n, 'e5_paired_permutation_S_lf_drop')}，均为后融合）与资源 × 方法 2×2（新资源只改变 {n.sd('e4_resource_gain_newmethod') if n.has('e4_resource_gain_newmethod') else '—'}）指向同一个结论：本轮的收益来自“把疾病名称、定义与 HPO 词条和基因的 GO 功能描述放进同一个冻结的生物医学语义空间，再学习一个有类型的匹配”，而蛋白序列与 HPA 表达在当前任务上只提供小的、部分可检测的补充。方案第 3.1 节把 GO 定义为功能注释视图而非独立原始模态；据此，本文把核心称为“多模态语义检索核心”，并明确“序列或表达带来独立增益”没有得到支持。",
         f"**为什么语义对齐有效，而 V6 的原始拼接无效。** V6 把 GO、通路与表达的 SVD 向量直接交给 LightGBM，树模型无法从“基因的一组功能向量”与“疾病的一组表型”之间学出成对的相似；V7.1 先在同一编码空间里把二者对齐，再把“匹配分数”交给二级模型，这正是原始拼接对照（{g(n, 'e2t_e2_raw_recall10_end')}）与后融合（{g(n, 'e2t_e2_lf_recall10_end')}）之间的差别。同资源零训练固定融合（{g(n, 'e1_srff_recall10_end')}）与线性双塔（{g(n, 'e2t_e2_tt_recall10_end')}）都更低，说明有类型投影与按模态分开的匹配分数是必要的；但功能锚在测试上没有显示独立收益（E2），不应把它当作本轮的有效机制。",
         f"**冷基因。** 核心把 V6 结构上不可达的答案带进名单，但后融合在挑选前十时仍偏向参考可见的基因：冷层前十恢复（{gp(n, 'e3_lflgb_unseen_top10')}、{gp(n, 'e3_lflgb_ppi_only_top10')}）低于核心单独（{gp(n, 'e3_core_unseen_top10')}、{gp(n, 'e3_core_ppi_only_top10')}），同时被挤出的旧命中约为新增命中的一半。分层校准或为冷基因单设融合权重是下一步最直接的改进点。严格基因监督冷变体与默认设置结果接近，说明冷基因的恢复不依赖它们自身的锚监督。",
         "**与已有路线的关系。** ProtST 与 BioTranslator 以文本—序列 / 文本—功能对齐为核心思想[43,44]，DeepPheno 由功能注释预测表型[45]；本文没有复现它们的完整软件与训练数据，而是在同一冻结编码器与同一评价口径下，以同资源固定融合、线性双塔、锚翻转与原始拼接四个可控对照检验“对齐是否必要”。本轮结果与这些工作的共同直觉一致——功能语义是连接疾病与基因的强桥梁——但也提示，在疾病关联稀缺的检索任务上，序列表示经功能锚的间接对齐并不自动带来额外收益。",
         "**评价边界。** 五折疾病已被 V5/V6 多轮设计使用，五折数字是内部估计；新确认集是本轮唯一预先冻结、从未触碰的标签集合，它的结果与五折方向一致，区间在零以上，但样本只有 618 个疾病。两项泄漏敏感性（二级文本掩蔽、2023 年版 GO）都只带来很小的变化；冻结编码器的预训练语料可能包含历史关联，本文的主声明是“当前监督与来源隔离”，而不是“基础模型从未见过”。",
         ("**关于题目。** 按登记的判定规则（后融合同时高于仅通道 LightGBM 与 F7、配对置换使性能下降、增益不由仅掩码模型或原始拼接复现），三项条件均满足，题目可以使用“多模态”；但结合 E4 与 E5，本文把它限定为“多模态语义检索核心”这一结构描述，而不写成“多种原始模态共同带来增益”。" if claim else
          "**关于题目。** 登记的“多模态增益”判定规则要求三项条件同时满足：" + "；".join(f"{k} = {v}" for k, v in crit.items()) + "。本轮不满足，因此题目不使用“多模态”，多模态核心作为已检验的扩展报告。")]
    return "\n\n".join(t)


def limitations(n) -> str:
    items = [
        "增益的主体来自疾病语义与 GO 功能语义的对齐；GO 与疾病—基因文献可能重叠。2023 年版 GO 的敏感性变化很小，但严格的历史实验还需要同时核验编码器、映射与全部输入的时间边界，本轮没有构建多模态核心的严格历史主表。",
        "不微调 SapBERT 与 ESM-2，不启用 ESMFold，也没有比较更大的 ESM；蛋白序列在后融合中的作用不显著，更强的序列表示能否改变这一点未检验。",
        "功能锚在开发集上有效，但在测试上没有显示独立收益；本文不把它作为已证实的机制。",
        "疾病组织条件来自登记的器官词别名表，不做否定、发育阶段与模糊器官描述的解析；宽泛规则只提供宽背景，不是临床组织定位。",
        "后融合的前十仍偏向参考可见基因，被挤出的旧命中约为新增命中的一半；冷基因的专门校准没有做。",
        "预测表型头、154 种细胞类型表达、图像—定位锚预训练与支持模态缺失训练属于后续增强，本轮没有实现；图像与小鼠通道在当前映射下为阴性结果。",
        "小鼠通道的 MP–HP 映射只做了程序化核验，人工抽查表已导出但未完成；敲除表型不能代表增益型或显性负效应机制，“未检测”与“检测正常”在断言文件中无法区分。",
        "ProtST、BioTranslator 与 DeepPheno 没有完整复现，本文以可控的同资源对照替代。",
        "人工双人盲审仍未完成；外部与时间任务沿用 V6 结论。",
        "五折结果是内部估计；新确认集的标签来自 GenCC 与 ClinVar，它们与 CTD 可能共享部分原始文献，过滤评价只排除了同一疾病的 CTD 阳性；小支持集合的最优核心权重落在网格上界。",
    ]
    return "## 6. 局限性\n\n" + "\n".join(f"{i + 1}. {x}" for i, x in enumerate(items))


def conclusion(n, claim) -> str:
    base = (f"V7.1 把“多模态设想”落实为可检查的计算流程：真实资源按主键对齐并记录来源，有类型核心以冻结编码器独立检索，"
            f"覆盖整条学习链的袋外后融合把新增候选转化为同一评价口径下的恢复——五折 Recall@10 由 V6 主模型的 {g(n, 'e1_v6lgb_recall10_end')} 提高到 {g(n, 'e1_lflgb_recall10_end')}，"
            f"预先冻结的新确认集上由 {g(n, 'e7_v6lgb_recall10')} 提高到 {g(n, 'e7_lflgb_recall10')}。")
    tail = ("配对置换、仅掩码模型与原始拼接对照表明增益依赖真实的输入对应；但增益的主体来自疾病语义与功能语义的对齐和后融合方法，蛋白序列与新表达资源的独立贡献很小，图像与跨物种表型未获支持。" if claim else
            "但按登记规则，多模态增益的证据尚不完整，本文把它作为已检验的扩展报告。")
    return "## 7. 结论\n\n" + base + tail + "V6 的全部阴性结果与本轮的有限、阴性结果一并保留。"

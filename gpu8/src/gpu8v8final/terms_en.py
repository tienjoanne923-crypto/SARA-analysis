"""English terminology of the SARA manuscript (text and tables); the figure wording is in figtext_en and uses the
same names.  ``GLOSSARY`` lists the Chinese -> English term pairs exported to the terminology table."""
from __future__ import annotations

from .figtext_en import MESH

# ------------------------------------------------------------------ methods (running-text form; cap() for table rows)
METHOD = {
    "deployed": "SARA", "channel_lgb_v6": "retrieval-channel LightGBM", "v71_direct_strict": "multimodal direct fusion",
    "v71_direct_v6": "multimodal direct fusion (retrieval top 500)", "B": "base ranker", "F7_strict": "fixed-weight fusion",
    "F7_v6": "fixed-weight fusion (retrieval top 500)", "standalone_sF": "semantic retrieval $s_F$",
    "standalone_r": "multimodal semantic core $r$", "standalone_sF0": "untrained SapBERT similarity",
    "standalone_sS": "semantic-model sequence score", "kdgene": "KDGene", "sign": "SIGN-style graph model",
}


def cap(s: str) -> str:
    return s[:1].upper() + s[1:] if s and not s.startswith(("SARA", "KDGene", "SIGN")) else s


def method(key: str, table: bool = False) -> str:
    s = METHOD.get(key, key)
    return cap(s) if table else s


def mesh(zh: str) -> str:
    """MeSH category name in running text (lower case), from its Chinese public name."""
    return MESH[zh][1]


def mesh_table(zh: str, code: str | None = None) -> str:
    s = cap(MESH[zh][1])
    return f"{s} ({code})" if code else s


# ------------------------------------------------------------------ ranker features (Table S2)
FAMILY = {"V6 检索通道": "Retrieval channels", "疾病描述": "Disease description", "候选融合": "Fusion rank",
          "语义": "Semantic", "表达": "Expression", "基因可用性": "Gene availability"}
SRC = {"pheno": "phenotype", "hier": "hierarchy", "onto": "ontology", "text": "text", "gene_hpo": "gene–phenotype",
       "rwr_pheno": "phenotype RWR", "compat": "compatibility"}
FEATURE = {**{f"v_{k}": f"{v} source score" for k, v in SRC.items()},
           **{f"lr_{k}": f"{v} source log rank" for k, v in SRC.items()},
           **{f"a_{k}": f"{v} source available" for k, v in SRC.items()},
           "fused": "retrieval fusion score", "ret_rank": "retrieval rank", "is_fill": "prior-fill flag", "prior": "degree prior",
           "d_hpo": "disease HPO term count", "d_pheno_sim": "disease phenotype similarity",
           "d_text_sim": "disease text similarity", "d_onto_ic": "ontology information content", "d_onto_inf": "ontology available",
           "dm_text": "disease has text", "dm_hpo": "disease has HPO", "dm_tissue": "disease has tissue",
           "lr_R": "fusion rank (availability-aware RRF)", "s_sem": "semantic score $s_F$", "lr_S": "semantic log rank",
           "m_sem": "semantic availability flag", "src_go": "semantic source = GO", "src_seq": "semantic source = sequence",
           "s_E": "expression score", "a_E": "expression available", "ref_visible": "reference edge visible",
           "hpo_visible": "gene–phenotype visible", "lg_ref_degree": "reference degree (log)", "gt_protein": "protein-coding",
           "gt_ncrna": "non-coding RNA", "has_seq": "has sequence", "has_go": "has GO", "has_expr": "has expression"}

# ------------------------------------------------------------------ experimental arms and task names
E4 = [("e4_layer1", "Training-visible source exclusion", "Training-visible source exclusion (mean of 5 folds)"),
      ("e4_layer1_random", "Matched random", None),
      ("e4_layer2", "Evaluation-conditioned deletion*", "Evaluation-conditioned deletion*"),
      ("e4_layer2_random", "Matched random*", "matched random*"),
      ("e4_blocks", "Evidence-block isolation", "Evidence-block isolation (GO part)"),
      ("e4_blocks_random", "Matched random", None),
      ("e4_experimental", "Experimental evidence only", "Experimental evidence only"),
      ("e4_no_tas_nas", "TAS/NAS removed", "TAS/NAS removed"),
      ("e4_no_mutant", "Mutant/genetic-interaction evidence removed", "Mutant/genetic-interaction removed"),
      ("e4_rescore_no_ic", "IC removed (re-scoring)", "IC removed"),
      ("e4_rescore_iea_only", "IEA only (re-scoring)", "IEA only")]
E5 = [("e5_none", "GO removed, no imputation"), ("e5_mlp", "Two-layer imputation model"), ("e5_linear", "Linear imputation model"),
      ("e5_nn", "Sequence-neighbor transfer"), ("e5_teacher_sS", "Semantic-model sequence projection"),
      ("e5_mlp_shuffled", "Two-layer imputation (pairs shuffled within length strata)"),
      ("e5_protst", "ProtST-input imputation (open-knowledge control)"),
      ("natural_student_on", "Imputation switched on, GO not removed")]
SUPPORT = {"joint_v8_w0.25": "SARA semantics + RWR", "joint_v6": "Baseline semantics + RWR",
           "qwalk_v6list": "Quantum walk (retrieval top 500)", "qwalk_v8list": "Quantum walk (SARA candidates)",
           "rwr_v6list": "RWR (retrieval top 500)", "rwr_v8list": "RWR (SARA candidates)", "sem_v6": "Baseline semantics alone",
           "sem_v8": "SARA semantics alone", "rwr": "RWR"}
EXPR_SET = [("v8_deployed_s42", "SARA (seed 42)"), ("v8_deployed_s52", "SARA (seed 52)"), ("v8_deployed_s62", "SARA (seed 62)"),
            ("v71_direct_strict_s42", "Multimodal direct fusion"), ("channel_lgb_v6_s42", "Retrieval-channel LightGBM"),
            ("F7_v6", "Fixed-weight fusion (retrieval top 500)"), ("matched_random", "Matched random gene sets")]
CONTRAST = {"lesional_vs_uninvolved_paired": "lesional vs non-lesional (paired)", "lesional_vs_control": "lesional vs control"}
SEMANTIC = {"教师 s_F（分段文本）": "Semantic retrieval $s_F$ (segmented text)", "V7.1 核心 r": "Multimodal semantic core $r$",
            "教师自身序列分数": "Semantic-model sequence score", "无学习 SapBERT 功能相似": "Untrained SapBERT similarity",
            "线性双塔": "Linear two-tower model", "ProtST 零样本": "ProtST zero-shot text–protein retrieval",
            "教师 s_F（V7.1 短文本）": "Semantic retrieval $s_F$ (truncated text)",
            "教师 s_F（二级掩蔽文本）": "Semantic retrieval $s_F$ (secondary masking)"}
HIST = {"ctd_relations": "CTD disease–gene associations", "ctd_evidence": "CTD literature evidence of associations",
        "ctd_vocabulary": "CTD disease vocabulary", "ctd_hierarchy": "CTD disease hierarchy", "hpo_obo": "HPO ontology",
        "hpoa": "HPO disease annotations", "hpo_phenotypes_mapped": "HPO disease mappings",
        "genes_to_phenotype": "Gene–phenotype annotations", "doid": "Human Disease Ontology", "go_obo": "GO ontology",
        "goa_human": "GOA human annotations", "hgnc": "HGNC gene nomenclature", "gtex_v8": "GTEx v8 tissue expression"}
LATER = {"gene background (19,861) and HGNC-based gene index": "gene background (19,861) and HGNC-based gene index",
         "protein sequences (MANE 2025)": "MANE representative protein sequences",
         "STRING v12.0 physical network": "STRING v12.0 physical interaction network",
         "disease equivalence groups and families": "disease equivalence groups and families",
         "V8.1 hyper-parameters and development choices": "SARA hyper-parameters and development-set choices",
         "SapBERT / ESM-2 checkpoints": "SapBERT and ESM-2 pretrained checkpoints"}
ROLE = {"entity index": "entity index", "static gene attribute": "static gene attribute",
        "PPI (not later than the cut-off)": "interaction network", "split units only": "used for splitting only",
        "frozen settings": "frozen settings", "frozen encoders": "frozen encoders"}
CASE = {"MESH:D024741": "familial hypertrophic cardiomyopathy", "MESH:D000544": "Alzheimer's disease",
        "MESH:D009765": "obesity", "MESH:D002471": "neoplastic cell transformation", "MESH:D012174": "retinitis pigmentosa",
        "MESH:D008113": "liver neoplasms"}

# ------------------------------------------------------------------ terminology table (Chinese -> English)
GLOSSARY = [
    ("方法与框架", [("SARA（可用性感知的语义检索与排序）", "SARA (Semantic retrieval with Availability-aware RAnking)"),
               ("知识覆盖不均衡", "uneven knowledge coverage"), ("疾病基因优先级排序", "disease gene prioritization"),
               ("候选生成 / 候选重排", "candidate generation / re-ranking"), ("多源检索通道", "multi-source retrieval channels"),
               ("疾病—功能语义模型", "disease–function semantic model"),
               ("倒数名次融合（RRF）", "reciprocal rank fusion (RRF)"), ("可用性感知 RRF", "availability-aware RRF"),
               ("普通 RRF", "standard RRF"), ("SARA 排序器", "SARA ranker"), ("SARA 候选", "SARA candidates"),
               ("检索前 500", "retrieval top 500"), ("扩展候选（约 950）", "extended candidates (~950)"),
               ("检索顺序 / 语义顺序", "retrieval order / semantic order"), ("语义分数 $s_F$", "semantic score $s_F$"),
               ("语义可用 / 语义不可用", "semantically available / unavailable"), ("融合名次", "fusion rank"),
               ("显式编码缺失值", "missingness-explicit (NaN) features"), ("功能锚", "functional anchors"),
               ("多正例排序损失", "multi-positive ranking loss"), ("正例—未标注学习", "positive–unlabeled learning")]),
    ("对照方法", [("检索通道 LightGBM", "retrieval-channel LightGBM"), ("基础排序器", "base ranker"),
              ("固定权重融合", "fixed-weight fusion"), ("多模态直接融合", "multimodal direct fusion"),
              ("语义检索 $s_F$", "semantic retrieval $s_F$"), ("多模态语义核心 $r$", "multimodal semantic core $r$"),
              ("SapBERT 零训练相似度", "untrained SapBERT similarity"), ("SIGN 式图神经网络", "SIGN-style graph model"),
              ("门控冷平衡纠偏器", "gated cold-balance corrector"), ("序列补位模型", "sequence imputation model")]),
    ("评价设计", [("家族隔离的嵌套交叉验证", "family-disjoint nested cross-validation"), ("外层折 / 内块", "outer fold / inner block"),
              ("上下文", "(fitting) context"), ("袋外（OOF）", "out-of-fold (OOF)"), ("开发集", "development set"),
              ("外部留出集", "external hold-out set"), ("参考疾病 / 参考关联", "reference diseases / reference associations"),
              ("参考路径冷", "reference-path cold"), ("冷启动阳性", "cold-start positives"),
              ("受控 GO 缺失", "controlled GO removal"), ("关联监督撤除（监督冷）", "association-supervision withdrawal (supervision cold)"),
              ("测试屏蔽集 / 开发屏蔽集", "test mask set / development mask set"), ("前 10 恢复率", "top-10 recovery"),
              ("可达上界 U@10", "attainable upper bound U@10"), ("覆盖@500", "coverage@500"),
              ("按疾病家族重抽样的 bootstrap", "family-level bootstrap")]),
    ("证据审计与外部检验", [("证据共享审计", "evidence-sharing audit"), ("训练可见来源排除", "training-visible source exclusion"),
                   ("评价条件化删除", "evaluation-conditioned deletion"), ("证据块隔离", "evidence-block isolation"),
                   ("匹配随机删除", "matched random deletion"), ("整链重训", "full-pipeline retraining"),
                   ("重打分", "re-scoring"), ("外部标签", "external labels"), ("过滤排名", "filtered ranking"),
                   ("文献独立子集", "literature-independent subset"), ("时间外推", "temporal extrapolation"),
                   ("新关联任务 / 新疾病任务", "new-association task / new-disease task"), ("支持任务", "support task"),
                   ("单样本", "one-shot"), ("表达队列", "expression cohorts")]),
]

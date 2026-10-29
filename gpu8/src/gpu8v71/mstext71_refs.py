"""New references (41-60) and result-independent text blocks of the V7.1 manuscript."""

REFS_V71 = [
    "Liu F, Shareghi E, Meng Z, Basaldella M, Collier N. Self-alignment pretraining for biomedical entity representations. NAACL-HLT. 2021:4228-4238.",
    "Lin Z, Akin H, Rao R, et al. Evolutionary-scale prediction of atomic-level protein structure with a language model. Science. 2023;379:1123-1130.",
    "Xu M, Yuan X, Miret S, Tang J. ProtST: multi-modality learning of protein sequences and biomedical texts. ICML. 2023:38749-38767.",
    "Xu H, Woicik A, Poon H, Altman RB, Wang S. Multilingual translation for zero-shot biomedical classification using BioTranslator. Nature Communications. 2023;14:738.",
    "Kulmanov M, Hoehndorf R. DeepPheno: predicting single gene loss-of-function phenotypes using an ontology-aware hierarchical classifier. PLOS Computational Biology. 2020;16:e1008453.",
    "Uhlén M, Fagerberg L, Hallström BM, et al. Tissue-based map of the human proteome. Science. 2015;347:1260419.",
    "Thul PJ, Åkesson L, Wiking M, et al. A subcellular map of the human proteome. Science. 2017;356:eaal3321.",
    "Morales J, Pujar S, Loveland JE, et al. A joint NCBI and EMBL-EBI transcript set for clinical genomics and research. Nature. 2022;604:310-315.",
    "The UniProt Consortium. UniProt: the Universal Protein Knowledgebase in 2023. Nucleic Acids Research. 2023;51:D523-D531.",
    "Groza T, Gomez FL, Mashhadi HH, et al. The International Mouse Phenotyping Consortium: comprehensive knockout phenotyping underpinning the study of human disease. Nucleic Acids Research. 2023;51:D1038-D1045.",
    "Baldarelli RM, Smith CL, Ringwald M, et al. Mouse Genome Informatics: an integrated knowledgebase system for the laboratory mouse. Genetics. 2024;227:iyae031.",
    "Smith CL, Eppig JT. The Mammalian Phenotype Ontology as a unifying standard for experimental and high-throughput phenotyping data. Mammalian Genome. 2012;23:653-668.",
    "Matentzoglu N, Balhoff JP, Bello SM, et al. A Simple Standard for Sharing Ontological Mappings (SSSOM). Database. 2022;2022:baac035.",
    "Smedley D, Oellrich A, Köhler S, et al. PhenoDigm: analyzing curated annotations to associate animal models with human diseases. Database. 2013;2013:bat025.",
    "Huang G, Liu Z, van der Maaten L, Weinberger KQ. Densely connected convolutional networks. CVPR. 2017:4700-4708.",
    "Oord A van den, Li Y, Vinyals O. Representation learning with contrastive predictive coding. arXiv:1807.03748. 2018.",
    "Khosla P, Teterwak P, Wang C, et al. Supervised contrastive learning. NeurIPS. 2020:18661-18673.",
    "Radford A, Kim JW, Hallacy C, et al. Learning transferable visual models from natural language supervision. ICML. 2021:8748-8763.",
    "Jacobs RA, Jordan MI, Nowlan SJ, Hinton GE. Adaptive mixtures of local experts. Neural Computation. 1991;3:79-87.",
    "Gu Y, Tinn R, Cheng H, et al. Domain-specific language model pretraining for biomedical natural language processing. ACM Transactions on Computing for Healthcare. 2021;3:2.",
]

RELATED_V71 = """多模态生物医学表示近年发展迅速。SapBERT 以 UMLS 同义词自对齐得到面向生物医学实体的语义表示[41,60]；ESM-2 以进化尺度蛋白语言模型给出序列表示[42]；ProtST 在蛋白序列与生物医学文本之间做多模态对齐[43]，BioTranslator 把自由文本描述翻译到基因、细胞与通路的零样本分类空间[44]，DeepPheno 以本体感知的层次分类器由功能注释预测基因缺失表型[45]。组织表达与亚细胞定位分别来自人类蛋白质图谱的组织与细胞图谱[46,47]；跨物种证据通常以 IMPC 实验表型[50]、MGI 同源关系与 MP 本体[51,52]以及 PhenoDigm/Exomiser 一类表型相似度连接人类疾病[34,54]，人工整理的本体映射以 SSSOM 标准共享[53]。对比学习与多正例损失[56–58]、门控专家混合[59]都是成熟工具。这些工作说明“把序列与文本对齐”“以门控组合模态”本身都不是原创；本文要检验的是：在疾病关联稀缺、候选覆盖是主要瓶颈的零支持任务里，冻结编码器上的一个小型有类型对齐核心能否独立检索到 V6 参考路径看不到的答案，其收益是否来自真实的模态对应而不是覆盖标志或研究热度，以及它与强通道基线合并后能否在同一评价口径下提高端到端恢复。ProtST、BioTranslator 与 DeepPheno 无法在本项目的输入与任务上完整复现，本文以同资源固定融合、线性双塔与原始拼接三个可控对照替代，并在讨论中说明改编范围。"""

INTRO_V71 = """第六轮（V6）把零支持候选恢复与支持驱动传播分开修复之后，端到端 Recall@10 从 0.059 提高到 0.109，但事后诊断同时给出了新的瓶颈：在当前参考路径中，约 14.9% 的答案关系只能经 PPI 邻居触达、8.0% 完全没有被表示，这部分答案几乎从不进入 500 候选；同一候选内 LightGBM 与调优融合几乎打平，而仅使用检索通道的 LightGBM 在事后分析中达到 0.118。换言之，V6 的剩余提升空间主要在“参考知识看不到的基因”上，而单纯给排序器再拼接几列特征（V6 已检验的 GO、通路与表达原始拼接）没有帮助。

第七轮（V7.1）据此回答一个更具体的问题：**能否用与 CTD 关联无关的原始模态——疾病语义、蛋白序列与正常组织表达——建立一个独立检索的核心，使它先在约两万个基因的完整背景上自己产生候选，再以袋外方式与 V6 互补？** 设计上有三点刻意的约束。第一，核心不读取 F7 或 LightGBM 的分数，必须独立产生候选，这样强旧通道不会掩盖新网络没有学到东西的事实。第二，每类输入有固定字段、维数、编码器与有效掩码，匹配分数按模态分别输出，“识别模态”不是让网络猜一列数字的来源。第三，“能运行”“门控非零”“梯度存在”都不被写成多模态有效：只有保留边际分布的配对置换使性能下降、并在同资源对照与强基线下仍有增益，才支持“模型用到了模态之间的对应”。"""

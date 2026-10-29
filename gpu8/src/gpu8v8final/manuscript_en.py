"""SARA manuscript in English (Briefings in Bioinformatics style): the English counterpart of manuscript10.

Numbers come from ``numbers9`` exactly as in the Chinese manuscript and are formatted by ``fmt_en`` (traced in
fmt9.TRACE); citations are ``[@key]`` markers resolved by ``refs_en`` into bracketed numbers; display equations are the
same LaTeX as in manuscript10 (so the same MathType objects are reused) and are cited as "Eq. (@eq:name)".
"""
from __future__ import annotations

import json
import re

import pandas as pd

from . import paths8
from .fmt_en import TRACE, ci, cin, i0, num, pc, rng, sci, sgn
from .manuscript10 import FIGS10, QQUAD, number_equations
from .manuscript9 import PUB, _meta, _row, acceptance_checks, table_md
from .manuscript9 import TABLE2_ORDER
from .numbers9 import ROOT, numbers9
from .terms_en import CASE, FEATURE, cap, mesh, method

DATE = "20260930"
TITLE = "SARA: availability-aware semantic retrieval of disease genes under uneven knowledge coverage"
EN = ROOT / "english"
STEM = f"SARA_manuscript_{DATE}"
ORD = {1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth", 6: "sixth"}


def mesh_pair(a: str, b: str) -> str:
    return f"{mesh(a)} and {mesh(b)}"


MATH_SPLIT = re.compile(r"(\$\$.*?\$\$|\$[^$\n]+\$)")


def typography(md: str) -> str:
    """typographic apostrophes in the text (primes inside LaTeX stay)."""
    parts = MATH_SPLIT.split(md)
    for i in range(0, len(parts), 2):
        parts[i] = re.sub(r"(?<=[A-Za-z])'(?=[A-Za-z])", "’", parts[i])
    return "".join(parts)


# ============================================================================ front matter
def front(R, N) -> str:
    z = N["z"]
    dep, ch = z["table"]["deployed"], z["table"]["channel_lgb_v6"]
    h = z["h"]
    eu = _row(pd.DataFrame(z["ext"]["contrasts"]), b="channel_lgb_v6", subset="union")
    cold_fold = dep["cold_recall10"] / ch["cold_recall10"]
    abstract = (
        "Disease–gene association knowledge is concentrated on a minority of well-studied genes, so candidates that lack "
        "known associations or functional annotations are systematically under-ranked by retrieval methods. SARA "
        "(Semantic retrieval with Availability-aware RAnking) addresses this imbalance with a two-stage framework that "
        "integrates multi-source retrieval, disease–function semantic retrieval and availability-aware ranking. In the first "
        "stage, an availability-aware reciprocal rank fusion selects 500 candidates per disease without penalizing candidates "
        "that lack a modality; in the second, a learning-to-rank model with explicitly encoded missing values re-ranks the "
        f"candidates. In family-disjoint five-fold cross-validation covering {i0(dep['n_diseases'])} test diseases from "
        f"{i0(h['h1_vs_channel_lgb_v6']['families'])} disease families, SARA reached a disease-macro-averaged top-10 recall "
        f"(Recall@10) of {num(dep['recall10'])}, an improvement of {ci(h['h1_vs_channel_lgb_v6'])} over a LightGBM ranker "
        f"using the same retrieval resources ({num(ch['recall10'])}), and the difference was positive in all 15 outer-fold "
        f"× seed combinations. Top-10 recovery of cold-start positives rose from {num(ch['cold_recall10'])} to "
        f"{num(dep['cold_recall10'])}, and the candidate budget needed to cover half of the test positives fell from about "
        f"{i0(N['K50_rB'])} to about {i0(N['K50_r_aware'])}. SARA performed comparably to a multimodal direct-fusion model "
        f"and showed a consistent direction of benefit on external labels for 618 diseases (difference {cin(eu)}), in a "
        "one-shot support task on 760 diseases and in four independent transcriptomic cohorts. TreeSHAP explanations "
        "identified semantic features as the main source of the rank gains of positives without a reference path, and a "
        "systematic evidence-sharing audit showed that the gains do not depend on functional annotations that share "
        "literature with the evaluated associations.")
    kp = [
        f"Disease–gene associations and functional annotations are concentrated on the same genes: the 10% of genes with "
        f"the most associations hold {pc(N['top10'], 0)} of all known associations and only {pc(N['all6'] / N['n_genes'])} "
        "of genes have all six information layers, so missing information is itself structural.",
        f"SARA generates candidates by availability-aware reciprocal rank fusion and re-ranks them with a missingness-"
        f"explicit ranker; under family-disjoint cross-validation it raised Recall@10 from {num(ch['recall10'])} to "
        f"{num(dep['recall10'])} and the recovery of cold-start positives about {cold_fold:.0f}-fold.",
        "The direction of benefit was consistent on external labels, in a one-shot support task and in independent "
        "transcriptomic cohorts; three-layer evidence deletion with matched random controls showed that the gains do not "
        "depend on shared literature.",
        "TreeSHAP explanations and stratified analyses show that semantic evidence provides the main positive "
        "contribution for candidates without a reference path, and that the gain grows with functional-annotation coverage.",
    ]
    return "\n\n".join([
        f"# {TITLE}",
        "**Running title:** SARA: availability-aware disease gene retrieval",
        "[Author name]¹, [Author name]¹ and [Corresponding author name]¹\\*",
        "¹[Department, institution, city, postcode, country]",
        "\\*Corresponding author: [Name], [postal address]. E-mail: [e-mail address]",
        "## Abstract", abstract,
        "**Keywords:** disease gene prioritization; uneven knowledge coverage; semantic retrieval; reciprocal rank fusion; "
        "availability-aware ranking; explainable machine learning",
        "## Key Points", "\n".join(f"- {k}" for k in kp),
    ])


# ============================================================================ 1 introduction
def intro(R, N) -> str:
    return "\n\n".join([
        "## 1 Introduction",
        "Identifying disease-associated genes underpins the elucidation of disease mechanisms, the interpretation of genetic "
        "variants and the discovery of therapeutic targets. Over two decades, disease gene prioritization has developed "
        "along several complementary lines: propagation and disease-module methods on protein–protein interaction (PPI) "
        "networks [@barabasi2011; @kohler2008; @vanunu2010; @ghiassian2015], integrative ranking by genomic data fusion "
        "[@aerts2006], clinical diagnostic tools driven by semantic similarity in the Human Phenotype Ontology (HPO) "
        "[@kohler2009; @smedley2015] and, more recently, representation learning on knowledge graphs and graph neural "
        "networks [@chandak2023; @huang2024txgnn; @wang2024kdgene; @saarinen2024]. These approaches share an implicit "
        "premise: evidence relevant to the target disease is comparable across candidate genes.",
        "Biomedical knowledge, however, accumulates very unevenly. A few genes attract most research attention, while many "
        "potentially important genes remain largely unexplored [@edwards2011; @stoeger2018]; functional annotations such "
        "as those of the Gene Ontology (GO) are likewise concentrated on well-studied genes [@haynes2018; @schnoes2013], "
        "and much of the performance of network methods rests on a few multifunctional ‘hub’ genes [@gillis2012]. In "
        "retrieval, this imbalance produces two systematic biases. First, genes without a reference association path are "
        "hard to reach for methods that propagate known associations. Second, candidates lacking a modality (for example, "
        "without GO or phenotype annotations) are often treated as poor matches rather than as ‘information "
        "unavailable’. Moreover, functional annotations and disease associations frequently derive from the same "
        "publications; if the evaluation does not isolate such shared evidence, reported gains may partly reflect "
        "information leakage [@kaufman2012; @cawley2010].",
        "Advances in biomedical and protein language models open a route around this imbalance [@lee2020biobert; "
        "@gu2021pubmedbert; @liu2021sapbert; @lin2023esm2; @xu2023protst]. With cross-modal alignment [@radford2021clip], "
        "the textual and phenotypic description of a disease can be compared directly with the functional description of "
        "a gene, reducing the dependence on known associations. The retrieve-then-rerank paradigm of information retrieval "
        "[@karpukhin2020; @nogueira2019], reciprocal rank fusion (RRF) [@cormack2009] and gradient-boosted learning to rank "
        "with LightGBM and LambdaRank [@ke2017; @burges2010] provide mature tools for integrating heterogeneous evidence "
        "under a limited candidate budget. Three questions nevertheless remain open: how to combine semantic evidence with "
        "conventional retrieval channels under a fixed candidate budget; how to distinguish a low score from missing "
        "information during modeling; and how to evaluate performance gains when shared evidence is rigorously isolated.",
        "To address these questions, this study introduces SARA (Semantic retrieval with Availability-aware RAnking). Its "
        "main contributions are threefold. (i) A disease–function semantic model maps disease text and HPO phenotypes into "
        "the semantic space of gene GO functions and thereby provides semantic evidence that does not rely on known "
        "associations. (ii) An availability-aware reciprocal rank fusion reduces exactly to the retrieval order for "
        "semantically unavailable candidates, so that they are not penalized, and merges the retrieval and semantic orders "
        "within a fixed budget of 500 candidates. (iii) A learning-to-rank model with explicitly encoded missing values "
        "re-ranks the candidates using retrieval, semantic and availability features generated out-of-fold (OOF), within an "
        "evaluation framework that combines family-disjoint cross-validation, three-layer evidence deletion, controlled "
        "annotation masking and several external tests.",
        "The remainder of the paper is organized as follows. Section 2 describes the data, the SARA framework and the "
        "evaluation methods; Section 3 reports the retrieval comparisons, the candidate-budget efficiency, the stratified "
        "and mechanistic analyses, the evidence-sharing audit and the external tests; Section 4 discusses the role of "
        "availability-aware design, the relation to existing methods and the limitations; and Section 5 concludes.",
    ])


# ============================================================================ 2 materials and methods
def methods(R, N) -> str:
    z = N["z"]
    M = _meta(N)
    au = z["audit"]
    l3 = au["layer3_evidence_blocks"]
    tx = json.loads((paths8.AUDIT / "text_segments_l1.json").read_text(encoding="utf-8"))
    mk = json.loads((paths8.MANIFESTS / "go_masks_v8.json").read_text(encoding="utf-8"))
    tr = json.loads((paths8.RUNS / "V8T__main__f0_outer__background__s42" / "result.json").read_text(encoding="utf-8"))
    cc, tc = tr["config"]["core"], tr["config"]["train"]
    acc = R["acceptance"]
    rep = z["repro"]
    hc = z["hist"]
    fl = z["ext"]["fields"]
    eu, ind = M["ext_union"], M["ext_indep"]
    so = fl["shared_original_evidence"]
    dev = z["dev"]
    sm = N["x"]["shap_meta"]
    n_checks, all_ok = acceptance_checks(R)
    assert all_ok
    return "\n\n".join([
        "## 2 Materials and methods",
        # ------------------------------------------------------------------ 2.1
        "### 2.1 Data sources and task definition",
        f"The candidate background comprised {i0(M['G'])} genes that map to the HUGO Gene Nomenclature Committee (HGNC) "
        f"[@seal2023]: {i0(M['protein'])} protein-coding genes, {i0(M['ncrna'])} non-coding RNAs (ncRNAs), "
        f"{i0(M['pseudo'])} pseudogenes and {i0(M['other'])} others. Genes lacking a sequence, GO annotations or expression "
        "data were retained in the background. Positive examples were the disease–gene associations of the marker/mechanism "
        f"type with direct evidence in the Comparative Toxicogenomics Database (CTD) [@davis2025], {i0(M['n_rel'])} in total "
        f"and covering {i0(M['dis_rel'])} diseases; therapeutic associations were not used as positives and served only for "
        "stratified evaluation. Disease-side information comprised names, definitions and synonyms (gene names masked by a "
        "fixed dictionary of HGNC symbols), HPO phenotype annotations mapped through strict equivalences [@gargano2024] "
        "(restricted to the phenotypic-abnormality branch), Medical Subject Headings (MeSH) tree numbers [@lipscomb2000], "
        "the Human Disease Ontology [@schriml2022] and Online Mendelian Inheritance in Man (OMIM) mappings "
        "[@amberger2019]. Gene-side information comprised GO annotations [@ashburner2000; @go2023], source-excluded "
        "gene–phenotype annotations, physical interactions with a confidence of at least 700 in STRING v12 "
        "[@szklarczyk2023], representative protein sequences [@morales2022; @uniprot2023] and consensus tissue expression "
        "from the Human Protein Atlas (HPA) [@uhlen2015]. Data volumes are listed in Table 1, and the distribution of "
        "information across genes and diseases is shown in Figure 1.",
        "The retrieval task is defined as follows: given the admissible text, phenotype and ontology information of a "
        "disease $d$ and the reference disease associations visible in the current context, retrieve the associated genes "
        "of $d$ from the whole gene background. To prevent information leakage, associations of the target disease and of "
        "its strictly equivalent diseases enter neither the reference association graph, nor the gene–phenotype sources, "
        "nor any second-level training feature; gene–phenotype annotations are aggregated to genes only after all diseases "
        "held out in the current context have been removed by source disease.",
        # ------------------------------------------------------------------ 2.2
        "### 2.2 Overview of the SARA framework",
        "SARA consists of two stages, candidate generation and candidate re-ranking (Figure 2a). In candidate generation, "
        "multi-source retrieval channels produce a retrieval order $r_B$ over all background genes from phenotypic, "
        "textual, ontological and hierarchical evidence (Section 2.4), and a disease–function semantic model maps diseases "
        "and genes into a shared semantic space, yielding a semantic score $s_F$ and a semantic order $r_S$ (Section 2.5). "
        "The two orders are combined by availability-aware reciprocal rank fusion, which keeps exactly 500 candidates per "
        "disease (Section 2.6). In re-ranking, the SARA ranker re-orders the candidates using 49 out-of-fold features "
        "(Section 2.7).",
        "The key property of the availability-aware fusion is that, when semantic information is unavailable, the fused "
        "score reduces exactly to the retrieval score, so a candidate without GO annotations is not additionally penalized "
        f"for the missing evidence. With the fusion weight $w=0.5$ in Eq. (@eq:aware), for example, a semantically "
        f"unavailable gene enters the candidate list whenever its retrieval rank is at most {i0(N['rrf_cut_aware'])}, "
        f"whereas standard RRF would require a retrieval rank of at most {i0(N['rrf_cut_plain'])} (Figure 2c). Three kinds "
        "of ‘cold’ state are distinguished (Figure 2d). A gene is reference-path cold if, in the evaluation context, it has "
        "neither a reference association nor a source-filtered gene–phenotype annotation; this state occurs naturally in "
        "the data, and the corresponding positives are called cold-start positives. The other two states, controlled GO "
        "removal and withdrawal of association supervision, are interventions designed to test the mechanism "
        "(Section 2.10).",
        # ------------------------------------------------------------------ 2.3
        "### 2.3 Family-disjoint nested cross-validation and out-of-fold features",
        "Each disease was assigned in advance to a single disease family (the third-level branch of its MeSH tree numbers; "
        "failing that, the branch of its nearest ancestor, or else its strict equivalence group), and all data splits were "
        "made at the family level. Diseases with positives were divided into non-overlapping development "
        f"({i0(M['dev'])} diseases), external hold-out ({i0(M['ext_heldout'])} diseases) and five outer folds (together "
        f"{i0(M['test'])} test diseases, {i0(M['test_rel'])} associations and {i0(M['test_fam'])} families). The $j$th "
        "outer context uses $F_j$ as the test fold, $F_{j+1}$ as the validation fold and the remaining three folds as the "
        "training set $T_j$, which is further split by family into three inner blocks $I_{jb}$; each fold therefore has "
        "three inner contexts and one outer context, 20 fitting contexts in total (Figure 2b).",
        "Every learned object — the semantic model, the retrieval channels, the candidate lists, the rankers and their "
        "normalization parameters — was generated out-of-fold within its own context: the second-level training features "
        "of inner block $I_{jb}$ come from models trained on $T_j\\backslash I_{jb}$, and the outer-context models are "
        "trained on $T_j$ and score $F_j$, $F_{j+1}$ and the development set. Exclusion rules follow the role of each "
        "disease in each context rather than its labels: the held-out set of an outer context is the union of $F_j$, "
        "$F_{j+1}$, the development set and the external hold-out set, and an inner context additionally holds out "
        "$I_{jb}$. Regenerating the retrieval orders and candidate lists after removing the test labels reproduced the "
        f"original results exactly, with a maximum difference in ranking scores of "
        f"{sci(acc['1_label_removal']['deployed_score_maxabs'])}.",
        # ------------------------------------------------------------------ 2.4
        "### 2.4 Multi-source retrieval channels",
        "For each disease, the retrieval channels compute scores from seven sources over all background genes of the "
        "current context, and reference diseases contribute only their visible associations [@kohler2008; @kohler2009]. "
        "The phenotype source measures the phenotypic proximity of diseases by an HPO similarity weighted by information "
        "content (IC). With $\\mathbf{q}_d$ the ancestor-closed HPO indicator vector of disease $d$, the information content "
        "of term $t$ and the similarity between diseases are",
        r"$$\mathrm{IC}(t)=-\ln\frac{n_t+1}{n_{\mathrm{ref}}+1},\qquad \mathrm{sim}(d,d')=\frac{\left(\mathbf{q}_d\odot\mathbf{c}\right)^{\mathrm{T}}\left(\mathbf{q}_{d'}\odot\mathbf{c}\right)}{\left\lVert \mathbf{q}_d\odot\mathbf{c}\right\rVert\,\left\lVert \mathbf{q}_{d'}\odot\mathbf{c}\right\rVert}$$ {#eq:ic}",
        "where $n_t$ is the number of reference diseases annotated with term $t$, $n_{\\mathrm{ref}}$ the number of "
        "reference diseases, $\\mathbf{c}$ the weight vector formed by $\\mathrm{IC}(t)$ and $\\odot$ the element-wise "
        "product. The phenotype-transfer source selects the 20 reference diseases $\\mathcal{N}_{20}(d)$ most similar to "
        "$d$ (negative similarities truncated to zero) and transfers their visible associations with normalized "
        "similarity weights:",
        r"$$s_d(g)=\sum\limits_{d'\in\mathcal{N}_{20}(d)}\frac{\mathrm{sim}(d,d')}{\sum\limits_{d''\in\mathcal{N}_{20}(d)}\mathrm{sim}(d,d'')}\times\frac{A_{d'g}}{\deg(d')}$$ {#eq:transfer}",
        "where $A_{d'g}$ indicates a reference association and $\\deg(d')$ is the number of visible associations of $d'$. "
        "The text and ontology sources select reference diseases by the cosine similarity of term frequency–inverse "
        "document frequency (TF-IDF) vectors of the gene-masked text and by the Resnik information content of the most "
        "informative common ancestor of the MeSH tree numbers, respectively, and transfer associations with "
        "Eq. (@eq:transfer); the hierarchy source uses the two-step neighborhood in the CTD disease hierarchy; the "
        "gene–phenotype source is the IC-weighted cosine similarity between the disease phenotype profile and the "
        "source-excluded phenotype profile of the gene; and the compatibility source is the cosine similarity between the "
        "disease phenotype profile and gene phenotype profiles aggregated from visible reference associations. The "
        "phenotype-propagation source runs a random walk with restart (RWR) on the STRING physical interaction network, "
        "seeded with the phenotype-transfer scores:",
        r"$$\mathbf{p}^{(t+1)}=\alpha\,\mathbf{u}+(1-\alpha)\,\mathbf{P}\,\mathbf{p}^{(t)}$$ {#eq:rwr}",
        "where $\\mathbf{u}$ is the normalized seed vector, $\\mathbf{P}$ the column-normalized interaction transition "
        "matrix and $\\alpha=0.15$ the restart probability; the iteration stops when "
        "$\\left\\lVert \\mathbf{p}^{(t+1)}-\\mathbf{p}^{(t)}\\right\\rVert_1\\le\\alpha\\varepsilon$ ($\\varepsilon=10^{-6}$). "
        "The ranks of each source over the whole background are combined by reciprocal rank fusion, and the weights of "
        "unavailable sources are redistributed per disease:",
        r"$$F_d(g)=\frac{\sum\limits_{s}w_s\,a_{ds}\,\phi_{ds}(g)}{\sum\limits_{s}w_s\,a_{ds}},\qquad \phi_{ds}(g)=\frac{1}{K_0+r_{ds}(g)}$$ {#eq:fusion}",
        "where $a_{ds}\\in\\{0,1\\}$ indicates whether source $s$ is available for disease $d$, $r_{ds}(g)$ is the rank of "
        "gene $g$ in source $s$ (genes not scored by the source have $\\phi_{ds}(g)=0$) and $K_0=60$. The fusion weights "
        "$w_s$ were selected on the development set by macro-averaged U@10, defined in Eq. (@eq:upper); the selection is "
        "described in Supplementary Note S2. Genes with a signal are ordered by $F_d(g)$ and the remaining genes by the "
        "degree prior, with ties broken by GeneID, which gives the full-background retrieval order $r_B$.",
        # ------------------------------------------------------------------ 2.5
        "### 2.5 Disease–function semantic model",
        "The semantic model encodes disease text, HPO terms and GO terms with the frozen biomedical entity-representation "
        "model SapBERT [@liu2021sapbert] and representative protein sequences with the frozen protein language model ESM-2 "
        "(35M-parameter version) [@lin2023esm2]. Disease text (name, definition and up to eight synonyms, all gene-masked) "
        "is split into segments at sentence boundaries, each of at most 126 word pieces, and segment vectors are averaged "
        f"with word-piece weights; in this way, the text of the {i0(tx['n_truncated_v71'])} diseases "
        f"({pc(tx['share_truncated_v71'])}) that exceed a single segment is encoded in full. The phenotype of a disease is "
        "represented by the annotation-frequency-weighted mean of its HPO term vectors and the function of a gene by the "
        f"aggregate of the vectors of its GO terms. A shared semantic projection $\\mathbf{{P}}_T$ (rank {cc['rank']}) maps "
        "disease text, disease phenotypes and gene functions into one space, and a second projection $\\mathbf{P}_S$ maps "
        "sequence representations into the same space; the match scores of the function, sequence and expression "
        "modalities are combined by a global gate into a core score $r(d,g)$. The training objective is the sum of a "
        "multi-positive ranking loss and a functional-anchor alignment loss:",
        r"$$\mathcal{L}_{\mathrm{DG}}=\frac{1}{\left|D_{\mathrm{tr}}\right|}\sum\limits_{d\in D_{\mathrm{tr}}}\frac{1}{\left|P_d\right|}\sum\limits_{g\in P_d}\ln\left(1+\sum\limits_{u\in U_d}\exp\frac{r(d,u)-r(d,g)}{\tau}\right)$$ {#eq:ldg}",
        r"$$\mathcal{L}_{\mathrm{A}}=\frac{1}{\left|G_A\right|}\sum\limits_{g\in G_A}\frac{1}{\left|A_g^{+}\right|}\sum\limits_{a\in A_g^{+}}\ln\left(1+\sum\limits_{b\in U_g}\exp\left(\ell_{gb}-\ell_{ga}\right)\right),\qquad \mathcal{L}=\mathcal{L}_{\mathrm{DG}}+\lambda_A\mathcal{L}_{\mathrm{A}}$$ {#eq:la}",
        f"where $P_d$ is the set of positive genes of training disease $d$ visible in the current context, $U_d$ the "
        f"unlabeled control genes (the full background minus the known positives of $d$) and $\\tau={tc['tau']}$ the "
        f"temperature; $\\ell_{{ga}}=\\left(\\mathbf{{z}}_g^{{S}}\\right)^{{\\mathrm{{T}}}}\\mathbf{{z}}_a/\\tau_a$ is the "
        f"similarity between the sequence representation of gene $g$ and GO functional anchor $a$ "
        f"($\\tau_a={tc['tau_a']}$), $A_g^{{+}}$ the anchors annotated to $g$ (GO terms annotated to 5–500 background "
        f"genes), $U_g$ the remaining anchors and $\\lambda_A={tc['lambda_A']}$. Unlabeled genes are not treated as "
        f"negatives; both losses follow the positive–unlabeled learning setting [@elkan2008; @bekker2020]. The model has "
        f"{i0(tr['fit']['n_params'])} trainable parameters and was trained with AdamW [@loshchilov2019] (learning rate "
        f"{tc['lr']}, weight decay {tc['weight_decay']}, {tc['batch']} diseases per batch) for at most {tc['max_epochs']} "
        f"epochs, with the checkpoint chosen by the full-background Recall@10 on $F_{{j+1}}$ of the same context "
        f"(patience {tc['patience']}); the unlabeled control set (reference-associated genes or the full background) was "
        "selected on the development set. SARA uses only the function-channel score, which does not depend on known "
        "associations:",
        r"$$\mathbf{z}_d=\mathrm{norm}\left(\mathbf{P}_T\left(\mathbf{x}_d^{\mathrm{text}},\mathbf{x}_d^{\mathrm{HPO}}\right)\right),\qquad \mathbf{z}_g^{F}=\mathrm{norm}\left(\mathbf{P}_T\,\mathbf{x}_g^{\mathrm{GO}}\right),\qquad s_F(d,g)=\mathbf{z}_d^{\mathrm{T}}\,\mathbf{z}_g^{F}$$ {#eq:sf}",
        "where $\\mathrm{norm}(\\mathbf{v})=\\mathbf{v}/\\left\\lVert\\mathbf{v}\\right\\rVert_2$ denotes L2 normalization; "
        "genes without GO annotations are semantically unavailable.",
        # ------------------------------------------------------------------ 2.6
        "### 2.6 Availability-aware reciprocal rank fusion",
        "Let $r_S(d,g)$ denote the semantic rank among semantically available genes and $m_{dg}\\in\\{0,1\\}$ the semantic "
        "availability flag. SARA fuses the retrieval and semantic orders as [@cormack2009]",
        r"$$b_{dg}=\frac{1}{K_0+r_B(d,g)},\qquad e_{dg}=\frac{m_{dg}}{K_0+r_S(d,g)},\qquad R(d,g)=\frac{w\,b_{dg}+(1-w)\,m_{dg}\,e_{dg}}{w+(1-w)\,m_{dg}}$$ {#eq:aware}",
        "If semantics are unavailable ($m_{dg}=0$), $R(d,g)=b_{dg}$, so a candidate is not penalized for a missing "
        "modality; if no gene has semantic information, the candidate list is identical to the retrieval top 500. For "
        "comparison, standard RRF has a constant denominator of 1, i.e. $w\\,b_{dg}+(1-w)\\,m_{dg}\\,e_{dg}$. For each "
        "disease, the 500 genes with the highest $R(d,g)$ form the candidate list (ties broken by GeneID), and no known "
        "positive is added. The weight $w\\in\\{0.3,0.5,0.7,0.9\\}$ was selected in each outer context by the positive "
        f"coverage on the development set (the larger value in case of ties); the five outer contexts selected "
        f"{', '.join(str(v) for v in dev['w'][:-1])} and {dev['w'][-1]}, and the positive coverage on the development set "
        f"rose from {pc(dev['v6_cov'])} for the retrieval top 500 to {pc(dev['strict_cov'])}.",
        # ------------------------------------------------------------------ 2.7
        "### 2.7 SARA ranker",
        "The SARA ranker is a LambdaRank learning-to-rank model [@burges2010] implemented in LightGBM [@ke2017] that "
        "re-orders the SARA candidates with 49 features: 30 retrieval features (scores, log ranks and availability flags of "
        "the seven sources; fusion score, retrieval rank, fill flag, degree prior and five disease-description statistics) "
        "and 19 semantic and availability features ($s_F$ and its log rank, the log rank of the fusion score $R$, semantic "
        "availability and source flags, the expression match score and its availability, reference-association and "
        "gene–phenotype visibility, reference degree, gene type and modality availability, and the availability of disease "
        "text, phenotype and tissue information). Missing semantic and expression features are entered explicitly as NaN "
        "rather than filled with zeros, so that the missing-value branches of the decision trees learn their meaning. "
        "The ranker was trained on the out-of-fold candidates and features of the three inner blocks with a "
        "learning rate of 0.05, at most 1,500 trees, early stopping on the Recall@10 of the outer $F_{j+1}$, and row and "
        "column subsampling of 0.8; six configurations combining $\\{15,31,63\\}$ leaves with $\\{50,200\\}$ minimum "
        "samples per leaf were compared on the development set (Supplementary Tables S1 and S2). Two alternative designs "
        "were also examined — a cold-balance corrector that adds a gated residual to the base ranking and a sequence "
        "imputation model that fills in missing GO annotations from protein language model representations; neither "
        "entered the final model (Supplementary Note S4).",
        # ------------------------------------------------------------------ 2.8
        "### 2.8 Comparison methods",
        "The comparison methods were: (i) retrieval-channel LightGBM, which uses the same 30 retrieval features as SARA "
        "and is trained on its own retrieval top 500; (ii) the base ranker, which uses the same 30 features and is trained "
        "out-of-fold on the SARA candidates; (iii) fixed-weight fusion, a linear fusion of the retrieval-source scores "
        "with fixed weights selected on the development set; (iv) multimodal direct fusion, retrained on the same SARA "
        "candidates with 14 multimodal semantic-core features added to the 30 retrieval features (missing values set to "
        "zero); (v) stand-alone full-background rankings by the semantic retrieval score $s_F$, the core score $r$ and the "
        "sequence score; (vi) untrained SapBERT disease–function similarity; and (vii) the knowledge-graph completion "
        "method KDGene [@wang2024kdgene] and a SIGN-style graph neural network [@frasca2020], retrained on the same "
        "candidate lists with checkpoints chosen on $F_{j+1}$ (Supplementary Table S3). In the support task, the "
        "comparison methods were an equally weighted random walk, the combination of semantics with the random walk and a "
        "continuous-time quantum walk [@saarinen2024].",
        # ------------------------------------------------------------------ 2.9
        "### 2.9 Functional-evidence policy and evidence-sharing audit",
        f"GO annotations were parsed from all 17 columns of the GAF 2.2 format; NOT qualifiers and ND evidence were removed "
        f"and evidence codes and references were kept, leaving {i0(M['go_rec'])} records under the production rules "
        f"({i0(M['has_go'])} genes with GO annotations). The evidence-sharing audit comprises three deletion layers: "
        "(i) training-visible source exclusion, which removes records citing the literature of the training associations "
        "of the current context; (ii) evaluation-conditioned deletion, which removes records citing the literature of any "
        "CTD association, including evaluation associations, and serves only as a sensitivity analysis; and (iii) "
        "evidence-block isolation, which draws connected components of the literature–association bipartite graph at "
        f"random (the giant component is never split) until {pc(l3['target_share'], 0)} of the associations are covered "
        "and isolates the associations in the blocks together with the GO records of their literature and the traceable "
        f"phenotype annotations ({i0(l3['blocks_chosen'])} blocks, {i0(l3['relations_held'])} associations, "
        f"{i0(l3['pmids_held'])} publications). Each layer has a matched random deletion control of the same size per "
        "gene; the evidence-code variants keep only experimental evidence, remove author-statement evidence (TAS/NAS), or "
        "remove mutant-phenotype and genetic-interaction evidence (IMP/IGI/HMP/HGI). Under every condition, the whole "
        "pipeline was retrained from scratch (seed 42), and the advantage of SARA over retrieval-channel LightGBM was "
        "compared before and after the deletion; removing curator-inferred evidence (evidence code IC) and keeping only "
        "electronic annotations (evidence code IEA) were run as re-scoring sensitivity analyses that replace only the "
        "inference inputs.",
        # ------------------------------------------------------------------ 2.10
        "### 2.10 Controlled removal of functional annotations and withdrawal of association supervision",
        f"Without reading any disease label, the {i0(mk['eligible_genes'])} genes with both GO annotations and a sequence "
        "were stratified by GO-term-count decile × physical-interaction-degree decile, and, in a stable hash order, 10% "
        f"were drawn as the development mask set ({i0(mk['G_mask_dev'])} genes) and the next 20% as the test mask set "
        f"({i0(mk['G_mask_test'])} genes); the two sets are disjoint. The GO annotations of masked genes and all quantities "
        "derived from them were removed before semantic-model training, functional-anchor construction and candidate "
        "generation. Withdrawal of association supervision (supervision cold) further removes all disease associations "
        "of the test-masked genes from the reference association graph, the gene–phenotype annotations and every "
        "training table, alone or combined with GO removal.",
        # ------------------------------------------------------------------ 2.11
        "### 2.11 External labels, temporal extrapolation, support tasks and expression cohorts",
        f"The external label set comprises GenCC [@distefano2022] (classification Definitive or Strong) and ClinVar "
        f"[@landrum2020] (review status of at least two stars) associations of {i0(fl['diseases'])} test diseases, "
        f"{i0(fl['pairs'])} in total and {i0(eu.get('n_pairs', 0))} pairs after merging and de-duplication. Each disease "
        "is evaluated with the outer model for which it is a test disease and has therefore never been a training disease; "
        "the CTD positives of the disease are removed from the ranking (filtered ranking), and results are averaged over "
        "three seeds. Every association is labeled by whether it shares source literature with a training association "
        f"or with the GO annotations of its gene (yes, {i0(so['yes'])}; no, {i0(so['no'])}; untraceable, "
        f"{i0(so['unknown'])}), and the literature-independent subset ({i0(ind.get('n_diseases', 0))} diseases, "
        f"{i0(ind.get('n_pairs', 0))} pairs) is reported separately.",
        "Temporal extrapolation uses 2023-12-31 as the cut-off. All inputs were rebuilt from the pre-cut-off versions of the "
        "CTD associations and disease vocabulary, HPO and gene–phenotype annotations, the Human Disease Ontology, GO/GOA, "
        f"HGNC and GTEx v8 expression [@gtex2020], all models were retrained, and two tasks were evaluated: new "
        f"associations of diseases already associated at the cut-off (new-association task, "
        f"{i0(hc['x2_deployed_vs_channel_lgb_v6']['n'])} diseases) and diseases without any association at the cut-off "
        f"(new-disease task, {i0(hc['x1_deployed_vs_channel_lgb_v6']['n'])} diseases); inputs dated after the cut-off are "
        "listed in Supplementary Table S9. The support task reveals $k\\in\\{0,1,3,5\\}$ known genes for 533 diseases, and "
        "the one-shot setting ($k=1$) is evaluated on 760 diseases; the combination weight $\\omega$ of semantics and the "
        f"random walk was selected on the development set ($\\omega={z['omega']}$).",
        "Expression validation used four cohorts from the Gene Expression Omnibus (GEO) [@barrett2013] for psoriasis "
        "(GSE13355, GSE30999) and ulcerative colitis (GSE87466, GSE75214). For each gene, the probe with the highest mean "
        "expression was kept, and differential expression was measured by a moderated $t$ statistic with variance "
        "shrinkage (following the empirical Bayes approach of limma [@ritchie2015]); the statistic ranks of the frozen top "
        "20/50 genes were compared with those of background genes matched on mean expression, physical-interaction degree "
        "and number of known associations, and random gene sets matched one-to-one on interaction degree and association "
        "count served as controls.",
        # ------------------------------------------------------------------ 2.12
        "### 2.12 Evaluation metrics and statistical inference",
        "The primary metric is the disease-macro-averaged top-$K$ recall, Recall@$K$; cold-start recovery is measured by the "
        "recall on cold-start positives:",
        r"$$R_d@K=\frac{\left|Q_d\cap\mathrm{Top}_K\left(C_d\right)\right|}{\left|Q_d\right|},\qquad R_d^{c}@K=\frac{\left|Q_d^{c}\cap\mathrm{Top}_K\left(C_d\right)\right|}{\left|Q_d^{c}\right|}$$ {#eq:recall}",
        "where $C_d$ is the candidate list of disease $d$, $Q_d$ its test associations and $Q_d^{c}$ the cold-start positives "
        "among them; $R^c$@$K$ is defined only for diseases with cold-start positives. To separate the contributions of "
        "candidate coverage and within-candidate ranking, the attainable upper bound of the candidate list is also "
        "reported:",
        r"$$U_d@K=\frac{\mathrm{min}\left(K,\,m_d\right)}{n_d}$$ {#eq:upper}",
        "where $m_d$ is the number of positives in the candidate list and $n_d$ the total number of positives; Hit@10 and "
        "the candidate coverage (coverage@500) are reported as well. The three random seeds (42, 52 and 62) are first "
        "averaged within each disease, and diseases without any positive among the candidates remain in the denominator. "
        "All confidence intervals (CIs) are bootstrap 95% percentile intervals obtained by resampling disease families "
        "[@efron1993; @field2007], with 10,000 resamples (seed 20260923) for the primary comparison and 1,000–2,000 for "
        "descriptive intervals; association-level proportions resample numerators and denominators jointly, with families "
        "as clusters. Apart from the primary comparison, only estimates and intervals are reported.",
        # ------------------------------------------------------------------ 2.13
        "### 2.13 Model explanation, visualization and implementation",
        "Model explanation used TreeSHAP, the exact tree-model algorithm of SHapley Additive exPlanations (SHAP) "
        "[@lundberg2017; @lundberg2020]: for the SARA ranker of outer fold 0 and seed 42, the per-feature contributions of "
        f"all held-out positives and {i0(sm['negatives_sampled'])} sampled unlabeled candidates were computed from the "
        "saved model and the rebuilt out-of-fold feature matrix, and the sum of the SHAP values was checked against the "
        "saved ranking scores. Distributions are drawn as Gaussian kernel density estimates (ranks on a log scale, with "
        "reflection at bounds), hierarchical clustering uses average linkage on Euclidean distances, source availability "
        "is shown as UpSet plots [@lex2014] and disease representations are visualized with t-SNE [@vandermaaten2008]; "
        "categorical palettes were checked by simulating color-vision deficiency [@machado2009], and each method keeps the "
        "same color in all figures.",
        f"All computations ran on a single workstation with Python {rep['environment']['python'].split()[0]}, PyTorch "
        f"{rep['environment']['torch'].split('+')[0]} (CUDA 12.8), LightGBM {rep['environment']['lightgbm']} and NumPy "
        f"{rep['environment']['numpy']} on one NVIDIA GeForce RTX 5060 Ti. All {i0(N['runs'])} fitting and evaluation runs "
        f"are recorded with content hashes; the timed runs took about {num(N['timed_h'], 1)} h in total, and the peak GPU "
        f"memory of a single run was {num(N['gpu_peak'] / 1024, 2)} GiB. In the same computing environment, the complete "
        "pipeline of fold 0 and seed 42 was recomputed from the frozen inputs with a fresh cache: the candidate lists and "
        "top 10 were identical, and the maximum difference in ranking scores was "
        f"{sci(rep['deployed_scores_maxabs'])}. A further {n_checks} whole-pipeline consistency checks all passed, covering "
        "label-removal invariance, filtering before aggregation, out-of-fold lineage, non-overlap of checkpoints and "
        "training sets, the integrity of GO masking and supervision withdrawal, the exactness of the candidate budget, the "
        "exact fallback when semantics are missing, batch invariance and seed rules.",
    ])


# ============================================================================ 3 results
def results(R, N) -> str:
    z, x = N["z"], N["x"]
    dep, ch = z["table"]["deployed"], z["table"]["channel_lgb_v6"]
    h = z["h"]
    lc, G = N["layer_counts"], N["n_genes"]
    bins = {b["known_relations"]: b for b in N["dis_bins"]}
    b0 = bins.get("0", {})
    jt = " and between ".join(f"{mesh_pair(a, b)} ({num(j, 2)})" for a, b, j, s in x["jac_top"][:2])
    rk = N["rk"]
    wa = _row(pd.DataFrame(N["wtl"]), stratum="全部")
    w5 = _row(pd.DataFrame(N["wtl"]), stratum="阳性数 >5")
    cold = {r["method"]: r for r in N["cold"]}
    st = z["strata"]
    mb, uq, sh = x["med_best"], x["uniq"], x["shared"]
    seeds = list(z["per_seed"].values())
    per_seed = ", ".join(num(v) for v in seeds[:-1]) + f" and {num(seeds[-1])}"
    parts = ["## 3 Results"]
    # ------------------------------------------------------------------ 3.1
    parts += [
        "### 3.1 Uneven coverage of disease–gene knowledge",
        f"Disease–gene knowledge is markedly concentrated on a minority of genes (Figure 1). Of the {i0(G)} background "
        f"genes, only "
        f"{i0(N['rel_genes'])} ({pc(N['rel_genes'] / G)}) have at least one known association; the 10% of genes with the "
        f"most associations account for {pc(N['top10'], 0)} of all known associations, and the Gini coefficients on the "
        f"gene and disease sides are {num(N['gini_g'], 2)} and {num(N['gini_d'], 2)} (Figure 1a–b). Association and "
        "annotation knowledge are concentrated on the same genes: the Spearman correlation between the number of "
        f"GO gene–term pairs of a gene and its number of known associations is {num(N['rho'], 2)}; "
        f"{'all' if N['go_rich_rel_share'] >= 1 else pc(N['go_rich_rel_share'])} genes with at least 100 GO gene–term "
        f"pairs already have known associations, whereas only {pc(N['go_poor_rel_share'])} of genes with 1–4 GO "
        "gene–term pairs do (Figure 1c).",
        f"The availability of the six information layers is similarly uneven (Figure 1d): {i0(lc['has_go'])} genes have GO "
        f"annotations, {i0(lc['has_expr'])} tissue expression and {i0(lc['has_ppi'])} physical interactions, but only "
        f"{i0(lc['has_gene_hpo'])} have gene–phenotype annotations; only {i0(N['all6'])} genes ({pc(N['all6'] / G)}) have "
        f"all six layers, whereas as many as {i0(N['no_rel_but_go_seq_expr'])} genes have GO annotations, a sequence and "
        "expression but no known association and can therefore be retrieved only through semantic information. Of the "
        f"{i0(N['ncrna_n'])} ncRNAs, {pc(N['ncrna_nogo_share'])} have no GO annotation. Diseases show the same gradient "
        f"(Figure 1e): {pc(b0.get('name_only'))} of the diseases without known associations have only a name and synonyms, "
        "whereas almost all diseases with more than 20 associations have a definition. Among the 12 MeSH categories with "
        f"the most associated genes, associated genes are widely shared, with the highest Jaccard coefficients between "
        f"{jt} (Figure 1f); knowledge coverage differs markedly across categories, and the share of associated diseases with "
        f"an HPO description ranges from {pc(x['hpo_lo'][1], 0)} for {mesh(x['hpo_lo'][0])} to "
        f"{pc(x['hpo_hi'][1], 0)} for {mesh(x['hpo_hi'][0])} (Figure 1g).",
        "These results show that uneven knowledge coverage is structural: a retrieval method that equates ‘missing "
        "information’ with ‘poor match’ systematically pushes down candidates that lack annotations or reference "
        "associations. This is the starting point of the availability-aware design of SARA; the overall retrieval "
        "performance of SARA relative to the baseline methods is evaluated first.",
    ]
    # ------------------------------------------------------------------ 3.2
    ks = (1, 5, 20, 50, 100)
    kvals = ", ".join(f"{num(rk['deployed'][k])} and {num(rk['channel_lgb_v6'][k])}" for k in ks)
    parts += [
        "### 3.2 Retrieval performance compared with baseline methods",
        f"On the {i0(dep['n_diseases'])} test diseases, the disease-macro-averaged Recall@10 of SARA was "
        f"{num(dep['recall10'])} ({per_seed} for the three seeds), an improvement of {ci(h['h1_vs_channel_lgb_v6'])} over "
        f"retrieval-channel LightGBM ({num(ch['recall10'])}), which uses the same retrieval resources and is trained on "
        "its own retrieval top 500 (Figure 3a, Table 2). The advantage was positive in all 15 outer-fold × seed "
        f"combinations, with the 95% CI excluding zero in {N['fs_ci_pos']} of them (range of differences "
        f"{rng(*N['fs_one_range'])}; Figure 3d), and it was stable across cut-offs: for $K$ = 1, 5, 20, 50 and 100, the "
        f"Recall@$K$ of SARA and retrieval-channel LightGBM were {kvals}, respectively (Figure 3c).",
        f"SARA also outperformed the base ranker that uses only retrieval features ({cin(h['vs_B'])}), fixed-weight fusion "
        f"({cin(h['vs_F7_v6'])}) and stand-alone semantic retrieval $s_F$ ({cin(h['vs_standalone_sF'])}), and it clearly "
        f"exceeded KDGene ({cin(z['comp']['kdgene']['deployed_minus'])}; seed 42) and the SIGN-style graph neural network "
        f"({cin(z['comp']['sign']['deployed_minus'])}; seed 42) retrained on the same candidate lists. Compared with "
        "multimodal direct fusion, which uses the semantic scores of all modalities (function, sequence and expression), "
        f"the difference was {ci(h['h2_vs_v71_direct_strict'])}, and the two methods performed comparably (Figure 3a, e).",
        f"Per-disease comparisons showed that SARA outperformed retrieval-channel LightGBM for {pc(wa['deployed_better'], 0)} "
        f"of the diseases and underperformed for {pc(wa['channel_better'], 0)}, the remainder being tied; among diseases "
        f"with more than five positives, improvements were more frequent ({pc(w5['deployed_better'], 0)} vs "
        f"{pc(w5['channel_better'], 0)}; Figure 3f). Measured by the best positive rank of each disease, the share of "
        f"diseases without any positive among the candidates fell from {pc(x['unl']['channel_lgb_v6'], 0)} for the "
        f"retrieval top 500 to {pc(x['unl']['deployed'], 0)} for the SARA candidates; on the same 500 candidates, adding "
        f"semantic and availability features moved the median best-positive rank from {num(mb['B'], 0)} for the base ranker "
        f"to {num(mb['deployed'], 0)} (Figure 3g; seed 42). Of the top-10 hits of SARA "
        f"({i0(x['hits_total'].get('deployed', 0))} disease–gene pairs), "
        f"{i0(sh.get(('deployed', 'v71_direct_strict'), 0))} were shared with multimodal direct fusion and "
        f"{i0(uq.get('deployed', 0))} were unique (Figure 3h); the attainable upper bound U@10 was {num(dep['U10'])} for "
        f"SARA and {num(ch['U10'])} for retrieval-channel LightGBM (Table 2).",
        f"Cold-start positives are recovered almost exclusively through semantic evidence (Figure 3b): on the "
        f"{i0(cold['deployed']['n_cold_diseases'])} diseases with cold-start positives, the $R^c$@10 of retrieval-channel "
        f"LightGBM and the base ranker was only {num(cold['channel_lgb_v6']['Rc10'])} and {num(cold['B']['Rc10'])}, "
        f"whereas SARA reached {num(cold['deployed']['Rc10'])}, about {dep['cold_recall10'] / ch['cold_recall10']:.0f} "
        "times higher. The advantage of SARA was broad: it led for single-association diseases "
        f"({num(st['deployed']['single_relation_recall10'])} vs {num(st['channel_lgb_v6']['single_relation_recall10'])}), "
        f"multi-association diseases ({num(st['deployed']['multi_relation_recall10'])} vs "
        f"{num(st['channel_lgb_v6']['multi_relation_recall10'])}), diseases with OMIM mappings "
        f"({num(st['deployed']['omim_mapped_recall10'])} vs {num(st['channel_lgb_v6']['omim_mapped_recall10'])}) and "
        f"diseases with more than five positives ({num(st['deployed']['burden_gt5'])} vs "
        f"{num(st['channel_lgb_v6']['burden_gt5'])}); on the {i0(st['deployed']['therapeutic_only_pairs'])} therapeutic "
        "associations that were never used for training, SARA also had the higher top-10 hit rate "
        f"({pc(st['deployed']['therapeutic_only_hit10_micro'])} vs {pc(st['channel_lgb_v6']['therapeutic_only_hit10_micro'])}). "
        "To identify the source of these gains, Section 3.3 separates the contributions of candidate generation and "
        "candidate re-ranking.",
    ]
    # ------------------------------------------------------------------ 3.3
    cv = N["cov500"]
    miss = pd.DataFrame(N["miss"])

    def mm(gr, li):
        return pc(_row(miss, group=gr, list=li).get("pool_share"))
    b = z["budget"]
    q, rs = x["quad"], x["resc"]
    hf = z["flows"]["channel_lgb_v6->deployed"]
    zs = z["zero"]
    br = z["var"]["bridge950"]["deployed"]
    parts += [
        "### 3.3 Budget efficiency of availability-aware candidate generation",
        f"The coverage curves as a function of the budget $K$ reveal the first source of the advantage of SARA "
        f"(Figure 4a). At $K=500$, the retrieval order covered {pc(cv['rB'][0])} of the test positives and the semantic "
        f"order {pc(cv['rS'][0])}, and their availability-aware fusion reached {pc(cv['r_aware'][0])}; the budget needed "
        f"to cover half of the test positives fell from about {i0(N['K50_rB'])} for the retrieval order to about "
        f"{i0(N['K50_r_aware'])}, a reduction of about {pc(1 - N['K50_r_aware'] / N['K50_rB'], 0)}. The overall "
        "coverage of standard and availability-aware RRF nearly coincided, but the two differed sharply for positives "
        "lacking semantics (Figure 4c): the share of ncRNA positives entering the candidates was "
        f"{mm('非编码 RNA', 'V6 500（仅通道）')}, {mm('非编码 RNA', '缺失感知严格 500')} and {mm('非编码 RNA', '普通 RRF 严格 500')} "
        "for the retrieval top 500, availability-aware RRF and standard RRF, respectively, and that of semantically "
        f"unavailable positives {mm('语义不可用', 'V6 500（仅通道）')}, {mm('语义不可用', '缺失感知严格 500')} and "
        f"{mm('语义不可用', '普通 RRF 严格 500')}. Availability-aware fusion thus largely preserved these candidates, "
        "whereas standard RRF halved their coverage.",
        f"Budget attribution splits the gain of SARA into two parts (Figure 4b, Table 3): on the same retrieval top 500, "
        f"the SARA ranker already improved on retrieval-channel LightGBM by {ci(b['v6pool_A_vs_channel'])}, and replacing "
        f"the candidates with the SARA candidates added a further {ci(b['strict_A_vs_v6pool_A'])}. Relaxing the budget to "
        f"about 950 candidates raised coverage to {pc(br['coverage500'])}; at this budget, the SARA ranker still exceeded "
        f"the base ranker by {ci(b['bridge_A_vs_B'])} but changed by only {ci(b['bridge_A_vs_strict_A'])} relative to 500 "
        "candidates, indicating that 500 candidates are an efficient budget close to saturation.",
        f"The joint distribution of the two orders shows where the semantic candidates come from (Figure 4f; "
        f"{i0(x['quad_n'])} semantically available test positives): {pc(q['均 ≤500'], 0)} of the positives entered the "
        f"top 500 of both orders and {pc(q['仅语义 ≤500'], 0)} only that of the semantic order; cold-start positives "
        "cluster around retrieval ranks of about $10^4$, and some of them are moved into the top 500 by the semantic order. "
        f"When the retrieval top 500 was replaced by the SARA candidates, {pc(rs['全部阳性']['语义救回'])} of the positives "
        f"were rescued by semantics and only {pc(rs['全部阳性']['被挤出'])} were displaced; among cold-start positives, "
        f"{pc(rs['参考路径冷']['语义救回'])} were rescued and only {pc(rs['参考路径冷']['被挤出'])} displaced (Figure 4g). "
        f"At the association level (Figure 4d–e), SARA moved {i0(N['flow']['flow_up'])} associations into a better rank "
        f"bin ({i0(N['flow']['flow_up_cold'])} of them cold-start positives); {i0(hf['kept'])} top-10 hits were kept and "
        f"{i0(hf['gained'])} gained ({i0(hf['gained_cold'])} of them cold-start genes). Another "
        f"{i0(zs['zero_signal_both'])} test associations ({pc(zs['zero_signal_share'])}) had no signal in either order, "
        "a common ceiling for retrieval methods based on existing knowledge. Beyond candidate generation, Section 3.4 "
        "examines how the gains of SARA are distributed across levels of knowledge coverage.",
    ]
    # ------------------------------------------------------------------ 3.4
    a5, b5, c5, d5 = N["s5a"], N["s5b"], N["s5c"], N["s5d"]

    def hv(df, stratum, m):
        return _row(df, stratum=stratum, method=m).get("hit10")
    hq = c5[(c5["hpo"]) & (c5["token_quartile"] >= 3)]["delta"]
    hq_lo = c5[(c5["hpo"]) & (c5["token_quartile"] >= 3)]["lo"]
    assert hq.min() >= c5["delta"].drop(hq.index).max()
    top = d5.sort_values("delta", ascending=False).head(3)
    t3 = [f"{mesh(r['name'])} ({sgn(r['delta'])}, 95% CI {rng(r['lo'], r['hi'])})" for _, r in top.iterrows()]
    t3 = "; ".join(t3[:-1]) + f"; and {t3[-1]}"
    pos = int((d5["lo"] > 0).sum())
    lf = sorted(x["lift"], key=lambda r: -r["mean_log10_lift"])
    ur = {(r["combo"] if isinstance(r["combo"], str) else f"{int(r['combo']):07d}"): r for r in x["up_rows"]}
    noh = ur.get("0111000", {})
    hqv = sorted(hq.tolist(), reverse=True)
    parts += [
        "### 3.4 Retrieval performance across levels of knowledge coverage",
        "Stratified by the number of reference associations of the gene in the evaluation context (Figure 5a), SARA "
        "exceeded retrieval-channel LightGBM at almost every level: association-level top-10 recovery of cold-start "
        f"positives rose from {pc(hv(a5, '冷', 'channel_lgb_v6'), 2)} to {pc(hv(a5, '冷', 'deployed'))}, and among genes "
        f"with at least 32 reference associations from {pc(hv(a5, '≥32', 'channel_lgb_v6'))} to "
        f"{pc(hv(a5, '≥32', 'deployed'))}. Stratified by GO annotation count (Figure 5b), the gain increased monotonically "
        f"with annotation richness, and SARA reached {pc(hv(b5, '≥100', 'deployed'))} among genes with at least 100 GO "
        f"gene–term pairs (retrieval-channel LightGBM, {pc(hv(b5, '≥100', 'channel_lgb_v6'))}). On the disease side "
        "(Figure 5c), diseases with an HPO description and a text length in the upper half benefited most (Recall@10 "
        f"differences of {sgn(hqv[0])} and {sgn(hqv[1])} in the two quartile cells"
        f"{', both 95% CIs excluding zero' if bool((hq_lo > 0).all()) else ''}).",
        f"Stratified by the availability pattern of the retrieval sources (Figure 2e), the {i0(x['up_total'])} test "
        f"diseases showed only {x['up_n_combo']} source combinations, and diseases with all seven sources accounted for "
        f"{pc(x['up_all7'] / x['up_total'], 0)}. Diseases without an HPO description lose four phenotype-related sources at "
        f"once, yet on the {i0(noh.get('n_diseases', 0))} diseases left with only the hierarchy, ontology and text sources, "
        f"SARA still improved on retrieval-channel LightGBM by {ci(noh)}; in {x['up_pos']} of the {x['up_n_combo']} "
        "combinations, the lower 95% CI bound exceeded zero.",
        f"Stratified by MeSH disease category (Figure 5d; {len(d5)} categories with at least 60 test diseases), the gain "
        f"of SARA over retrieval-channel LightGBM had a lower 95% CI bound above zero in {pos} categories, and the three "
        f"largest gains were in {t3}. Measured by the log improvement of the best positive rank of each disease "
        f"(Figure 5e), the mean improvement was positive in all {len(lf)} categories and largest for {mesh(lf[0]['name'])} "
        f"(on average {num(lf[0]['mean_log10_lift'], 2)} orders of magnitude higher, with the best rank improved for "
        f"{pc(lf[0]['better'], 0)} of the diseases). These strata show that the gains of SARA are not confined to "
        "particular diseases or genes but grow with the available semantic evidence; this pattern points to the way in "
        "which the ranker uses semantic evidence, which Section 3.5 examines through model explanation.",
    ]
    # ------------------------------------------------------------------ 3.5
    fi = N["fi"].sort_values("mean", ascending=False)
    fam = N["fi_fam"]
    stp, sf, sm = x["shap_top"], x["shap_fam"], x["shap_meta"]
    ca, nc = sf["阳性·参考路径冷"], sf["阳性·非冷"]
    labs = [FEATURE[r["feature"]] for r in stp]
    items = [(f"{lb[:-1]}; {num(r['mean_abs_shap'])})" if lb.endswith(")") else f"{lb} ({num(r['mean_abs_shap'])})")
             for lb, r in zip(labs, stp)]
    top_txt = ", ".join(items[:-1]) + f" and {items[-1]}"
    eo, er = z["var"]["expr_off"]["deployed"], z["var"]["expr_random"]["deployed"]
    for v in (eo, er):
        assert v["delta_vs_main"]["ci_low"] <= 0 <= v["delta_vs_main"]["ci_high"]
    e_rank = next(i for i, r in enumerate(stp) if r["feature"] == "s_E") + 1
    df_, dg_ = x["dep_f"], x["dep_g"]
    lab0 = FEATURE[str(fi.iloc[0]["feature"])]
    parts += [
        "### 3.5 Feature contributions and interpretability of the ranker",
        f"Across the 15 ranking models (5 outer folds × 3 seeds), the single most important feature of the SARA ranker was "
        f"the {lab0}, with {pc(fi.iloc[0]['mean'])} of the feature gain; summed by feature family, the retrieval channels "
        f"accounted for {pc(fam.get('V6 检索通道', 0), 0)}, the fusion rank for {pc(fam.get('候选融合', 0), 0)} and the "
        f"semantic features for {pc(fam.get('语义', 0), 0)} (Figure 6a). TreeSHAP provides per-candidate attributions "
        f"(Figure 6b–e; outer fold 0, seed 42, {i0(sm['positives'])} held-out positives and "
        f"{i0(sm['negatives_sampled'])} unlabeled candidates; the maximum difference between the summed SHAP values and "
        f"the model output was {sci(sm['reconstruction_maxabs_vs_saved_scores'])}). The features with the largest mean "
        f"absolute SHAP values were the {top_txt} (Figure 6c). The contribution of the fusion rank decreased almost "
        f"monotonically with rank, from {sgn(df_['first'], 2)} at a rank of about {num(df_['first_rank'], 0)} to "
        f"{sgn(df_['last_pos'], 2)} at about {num(df_['last_pos_rank'], 0)}, and became negative thereafter (Figure 6d); the "
        f"contribution of the semantic score $s_F$ increased steeply above about {num(dg_['zero_hi'], 2)} and reached "
        f"{sgn(dg_['top'], 2)} at an $s_F$ of about {num(dg_['top_x'], 2)} (Figure 6e).",
        "The signed contributions of the feature families summed by candidate class (Figure 6b) show that non-cold "
        f"positives are pushed up mainly by the retrieval channels ({sgn(nc['V6 检索通道'], 2)}), with a semantic "
        f"contribution of {sgn(nc['语义'], 2)}, whereas cold-start positives gain mainly from the semantic family "
        f"({sgn(ca['语义'], 2)}) and the fusion rank ({sgn(ca['候选融合'], 2)}), the contribution of the retrieval channels "
        f"being {sgn(ca['V6 检索通道'], 2)}. In other words, SARA follows retrieval evidence where it is sufficient and lets "
        "semantic evidence take over where it is missing, which is the behavior intended by the availability-aware design. "
        f"The expression score ranked {ORD[e_rank]} by mean absolute SHAP value, but turning off all expression features or "
        f"replacing them with random tissue expression gave Recall@10 of {num(eo['recall10'])} and {num(er['recall10'])}, "
        f"with 95% CIs of the difference from the full model ({num(eo['delta_vs_main']['b'])}) including zero in both "
        "cases (Figure 6f, Table 3); expression information is therefore highly redundant with the other features, and "
        "the performance of SARA does not depend on it. Because semantic evidence derives mainly from GO functional "
        "annotations, Section 3.6 tests its causal role with controlled masking.",
    ]
    # ------------------------------------------------------------------ 3.6
    e6 = N["e6"]

    def g(arm, col):
        return float(e6.loc[arm, col])
    parts += [
        "### 3.6 Effect of functional annotation and association supervision on cold-start gene retrieval",
        "To test the source of semantic evidence, 20% of the genes were drawn as the test mask set within deciles of GO "
        "term count × physical-interaction degree without reading any disease label; the distributions of masked and "
        "unmasked genes on both measures nearly coincide (Figure 7a). After all GO annotations of these genes were removed, "
        f"the share of their positives entering the SARA candidates fell from {pc(g('main', 'mask_pool'))} to "
        f"{pc(g('e5_none', 'mask_pool'))} and the share entering the top 10 from {pc(g('main', 'mask_hit10'))} to "
        f"{pc(g('e5_none', 'mask_hit10'))}, while the overall Recall@10 over all diseases changed by only "
        f"{sgn(g('e5_none', 'overall_delta'))} (Figure 7b). The other channels of SARA thus carry most of the retrieval "
        "task, and the contribution of the semantic channel is concentrated on genes with functional annotations. Rank "
        f"flows of the same positives (Figure 7d) show that, of the {i0(x['flow_main_top10'])} masked-gene positives in the "
        f"top 10 of the full model, {i0(x['flow_main_top10_kept'])} remained in the top 10 after GO removal.",
        "A stricter test is the withdrawal of association supervision: all known associations of the masked genes were "
        "removed from the reference association graph, the gene–phenotype annotations and every training table, so that "
        "these genes were completely invisible during training. With GO functional semantics alone, SARA still brought "
        f"{pc(g('e6_cold', 'mask_pool'))} of their positives into the candidates and {pc(g('e6_cold', 'mask_hit10'))} into "
        f"the top 10; when GO annotations were also removed, these shares fell to {pc(g('e6_cold_gomask', 'mask_pool'))} "
        f"and {pc(g('e6_cold_gomask', 'mask_hit10'))} (Figure 7b–d). For genes entirely lacking association supervision, "
        "functional annotations therefore provide the main retrieval signal through the disease–function semantic model "
        "and are the only route into the top 10. Several ways of imputing missing GO annotations from protein language "
        "model representations were also examined (Supplementary Note S4, Supplementary Figure S8); with the current "
        "data, experimentally observed functional annotations remain an irreplaceable source of information. Because "
        "functional annotations and disease associations may derive from the same publications, Section 3.7 tests whether "
        "this dependence amounts to information leakage.",
    ]
    # ------------------------------------------------------------------ 3.7
    au = z["audit"]
    l3 = au["layer3_evidence_blocks"]
    codes = N["codes"].copy()
    tot_no = codes["no_pmid"].sum() / codes["total"].sum()
    iea_iba = codes[codes["evidence_code"].isin(["IEA", "IBA"])]["no_pmid"].sum() / codes["no_pmid"].sum()
    e4, e4s = z["e4"], z["e4s"]
    full = e4s["full_gap"]
    nm = e4["e4_no_mutant"]
    keep = nm["new_minus_old_reduced"]["delta"] / full["delta"]
    parts += [
        "### 3.7 Influence of evidence sharing on the performance gain",
        "Functional annotations and disease associations may originate from the same publications, so the GO evidence was "
        f"audited systematically (Figure 8). Of the GO records retained by the production rules, {pc(tot_no)} lack a PubMed "
        f"identifier (PMID), {pc(iea_iba, 0)} of them electronic (IEA) or phylogenetic (IBA) annotations; these annotations "
        "were treated as being of unknown source (Figure 8a). The literature–association bipartite graph has "
        f"{i0(l3['n_components'])} connected components with a median size of "
        f"{num(l3['component_size_quantiles']['0.5'], 0)} (Figure 8b). The audit comprises three deletion layers, each with "
        "a matched random deletion control of the same size per gene: (i) removal of GO records citing the literature of "
        "the training associations of the current context; (ii) removal of GO records citing the literature of any "
        "evaluation association (a sensitivity analysis not used in any model selection); and (iii) isolation of "
        f"{i0(l3['relations_held'])} associations together with their literature, as whole connected components; three "
        "evidence-code variants were added. Under every condition, the whole pipeline was retrained from scratch "
        "(Figure 8c).",
        f"In {e4s['n_inc0']} of the {e4s['n_retrain']} deletion conditions with full-pipeline retraining, the 95% CI of the "
        "change in the advantage of SARA over retrieval-channel LightGBM included zero, and the changes resembled those of "
        "the matched random deletions (Figure 8d); even removing the GO records that cite the literature of any evaluation "
        f"association changed the advantage by only {ci(e4['e4_layer2']['difference'])}. The only condition whose 95% CI "
        "excluded zero was the removal of mutant-phenotype and genetic-interaction evidence (IMP/IGI/HMP/HGI): the "
        f"advantage changed by {ci(nm['difference'])}, about {pc(abs(nm['difference']['delta']) / full['delta'], 0)} of the "
        f"full advantage of {sgn(full['delta'])}, and SARA retained {pc(keep, 0)} of its advantage, consistent with such "
        "experimental evidence characterizing gene function directly. Broken down by disease category (Figure 8e), "
        f"{pc(x['dod_ret_small'], 0)} of the ‘category × condition’ cells under full-pipeline retraining changed by at most "
        "0.02, and targeted deletions agreed in sign with their matched random controls within the same category in "
        f"{pc(min(x['dod_sign_agree']), 0)}–{pc(max(x['dod_sign_agree']), 0)} of the categories, close to chance, indicating "
        "that differences between categories mainly reflect sampling variation. Re-scoring with electronic annotations "
        f"substituted only at inference changed the advantage by {ci(e4['e4_rescore_iea_only']['difference'])}, which "
        "reflects the sensitivity of a model trained on complete annotations to an abrupt shift of its input distribution "
        "rather than a gain from shared literature. Overall, the gain of SARA stems from the biological information "
        "carried by the functional annotations themselves. With information leakage ruled out, Section 3.8 examines how "
        "SARA performs outside the training distribution.",
    ]
    # ------------------------------------------------------------------ 3.8
    ext = pd.DataFrame(z["ext"]["contrasts"])

    def ec(bb, sub):
        return ci(_row(ext, b=bb, subset=sub))
    fl = z["ext"]["fields"]
    et = pd.DataFrame(z["ext"]["table"])
    eu = _row(et, method="deployed", subset="union")
    ei = _row(et, method="deployed", subset="independent_evidence")
    hc = z["hist"]
    x2 = x["hp"]["x2"]
    sup = {(r["method"], r["k"]): r for r in z["support533"]}
    s1 = {r["method"]: r for r in z["support1"]}
    sem = N["sup"]
    k0v8 = _row(sem, method="sem_v8", k=0).get("recall10")
    k0v6 = _row(sem, method="sem_v6", k=0).get("recall10")
    sp1 = x["sup_pairs"].get(1, {})
    ex = z["expr"]
    parts += [
        "### 3.8 External labels, temporal extrapolation and application scenarios",
        f"SARA was first tested on external labels independent of CTD: {i0(eu.get('n_pairs', 0))} GenCC and ClinVar "
        f"associations of {i0(fl['diseases'])} test diseases, none of which had been a training disease of the model that "
        f"evaluated it. SARA improved on retrieval-channel LightGBM by {ec('channel_lgb_v6', 'union')}, with "
        f"{ec('channel_lgb_v6', 'GenCC')} on the GenCC subset and {ec('channel_lgb_v6', 'ClinVar')} on the ClinVar subset, "
        f"and on the base ranker by {ec('B', 'union')} (Figure 9a). On the subset without source literature shared with "
        f"training associations ({i0(ei.get('n_diseases', 0))} diseases, {i0(ei.get('n_pairs', 0))} pairs), SARA improved "
        f"on the base ranker by {ec('B', 'independent_evidence')}, and the difference from retrieval-channel LightGBM was "
        f"{ec('channel_lgb_v6', 'independent_evidence')}, in the same direction; SARA and multimodal direct fusion "
        "performed comparably on all subsets (Figure 9a–b).",
        "In temporal extrapolation with a 2023-12-31 cut-off, in which all models were retrained with pre-cut-off "
        "resources, SARA and retrieval-channel LightGBM performed comparably on the new-association task "
        f"({cin(hc['x2_deployed_vs_channel_lgb_v6'])}) and likewise on the new-disease task "
        f"({cin(hc['x1_deployed_vs_channel_lgb_v6'])}), whereas SARA outperformed stand-alone semantic retrieval on the "
        f"new-disease task ({cin(hc['x1_deployed_vs_standalone_sF'])}); pairwise comparison showed that SARA brought "
        f"slightly more new associations into the candidates ({i0(x2['listed_deployed'])} vs {i0(x2['listed_channel'])} "
        "pairs on the new-association task; Figure 9c, f; Supplementary Note S5).",
        "In application scenarios with a few known support genes, the semantic representation of SARA was equally "
        "effective (Figure 9d, g). In the support task on 533 diseases, SARA semantics alone reached a Recall@10 of "
        f"{num(k0v8)} without support genes, above the {num(k0v6)} of the baseline semantics; with five support genes "
        f"revealed, the combination of SARA semantics with RWR reached {num(sup[('joint_v8_w0.25', 5)]['recall10'])}, above "
        f"the baseline semantic combination ({num(sup[('joint_v6', 5)]['recall10'])}), RWR alone "
        f"({num(sup[('rwr_v8list', 5)]['recall10'])}) and the quantum walk ({num(sup[('qwalk_v8list', 5)]['recall10'])}) "
        f"[@saarinen2024]; in per-disease comparisons with one support gene revealed, SARA was better for "
        f"{i0(sp1.get('up', 0))} diseases and worse for {i0(sp1.get('down', 0))}. In the larger one-shot task on 760 "
        f"diseases, the combination of SARA semantics with RWR reached a Recall@10 of {num(s1['joint_v8_w0.25']['recall10'])}, "
        f"above the {num(s1['joint_v6']['recall10'])} of the baseline semantic combination and the "
        f"{num(s1['rwr']['recall10'])} of RWR alone (Recall@50: {num(s1['joint_v8_w0.25']['recall50'])}, "
        f"{num(s1['joint_v6']['recall50'])} and {num(s1['rwr']['recall50'])}).",
        "Finally, the frozen top-ranked gene sets were tested in four independent transcriptomic cohorts (two for "
        "psoriasis [@gudjonsson2010; @suarezfarinas2012] and two for ulcerative colitis [@li2018uc; @vancamelbeke2017]). In "
        "10 rank tests (cohort × contrast × gene-set size), the top genes of the three SARA seeds were significantly "
        f"enriched ($P<0.05$) in {ex['v8_deployed_s42']['p05']}, {ex['v8_deployed_s52']['p05']} and "
        f"{ex['v8_deployed_s62']['p05']} tests, those of multimodal direct fusion in {ex['v71_direct_strict_s42']['p05']}, "
        f"and those of retrieval-channel LightGBM and the matched random gene sets in {ex['channel_lgb_v6_s42']['p05']} "
        "(Figure 9e, Supplementary Table S10). These results provide independent transcriptomic support for the disease "
        "relevance of the top-ranked genes of SARA; Section 3.9 illustrates their biological meaning with specific "
        "diseases.",
    ]
    # ------------------------------------------------------------------ 3.9
    rows = N["case_rows"]
    cs = []
    for c in N["cases"][:4]:
        d = c["disease_id"]
        rr = rows[rows["disease_id"] == d]
        cs.append(f"{int(rr[(rr['method'] == 'deployed')]['held_out_positive'].sum())} and "
                  f"{int(rr[(rr['method'] == 'channel_lgb_v6')]['held_out_positive'].sum())} for {CASE[d]}")
    nets = {n_["disease_id"]: n_ for n_ in x["nets"]}
    hcm, ad = nets.get("MESH:D024741", {}), nets.get("MESH:D000544", {})
    rp = rows[rows["disease_id"] == "MESH:D012174"]
    rp_d = int(rp[(rp["method"] == "deployed")]["held_out_positive"].sum())
    rp_c = int(rp[(rp["method"] == "channel_lgb_v6")]["held_out_positive"].sum())
    parts += [
        "### 3.9 Case studies of representative diseases",
        "Six test diseases were selected by a rule fixed before plotting (Figure 10a): among diseases with at least three "
        "held-out positives, the disease with the largest top-10 gain of SARA in each of four categories, plus the two "
        "diseases with the smallest gains overall. In the four gain cases, the numbers of held-out positives in the top 10 "
        "of SARA and retrieval-channel LightGBM were " + "; ".join(cs[:-1]) + f"; and {cs[-1]}.",
        "The positives recovered by SARA form functionally coherent modules. In familial hypertrophic cardiomyopathy, the "
        "top 10 of SARA concentrate on sarcomeric proteins (MYH6, MYH7, TNNT2, TNNI3, MYL2 and others), with "
        f"{i0(hcm.get('phys_dep', 0))} STRING physical interactions among them [@szklarczyk2023], TNNI3 and MYL2 being "
        "cold-start positives; in Alzheimer's disease, SARA added APOE, APP and CLU to the PRNP, MAPT, GSK3B and PSEN1 "
        f"shared by both methods, raising the number of physical interactions within the top 10 from {i0(ad.get('phys_ch', 0))} "
        f"to {i0(ad.get('phys_dep', 0))} (Figure 10b). For diseases in which the retrieval channels are already strong (for "
        f"example retinitis pigmentosa, with {rp_d} and {rp_c} held-out positives in the top 10), the top 10 of SARA still "
        "concentrate on photoreceptor and retinal genes (PROM1, RP1, REEP6, GUCA1A and opsin genes), several of which are "
        "not yet annotated as associated with the disease, suggesting that semantic and retrieval evidence are "
        "complementary.",
    ]
    return "\n\n".join(parts)


# ============================================================================ 4 discussion, 5 conclusion
def discussion(R, N) -> str:
    z = N["z"]
    dep, ch = z["table"]["deployed"], z["table"]["channel_lgb_v6"]
    fam = N["fi_fam"]
    ca = N["x"]["shap_fam"]["阳性·参考路径冷"]
    comp = z["comp"]
    return "\n\n".join([
        "## 4 Discussion",
        "### 4.1 Role of availability-aware design in cold-start retrieval",
        "The difficulty of disease gene retrieval lies not only in model capacity but in the unevenness of knowledge "
        "itself: known associations, functional annotations and phenotype descriptions are concentrated in the same "
        "direction, which places genes that lack them at an inherent disadvantage. SARA addresses this at three levels: a "
        "disease–function semantic model provides evidence that does not depend on known associations; availability-aware "
        "reciprocal rank fusion merges the retrieval and semantic orders within a fixed budget while ensuring that "
        "candidates missing a modality are not additionally penalized; and a learning-to-rank model with explicitly "
        "encoded missing values re-ranks the candidates on out-of-fold features. Under strict family-disjoint evaluation, "
        f"SARA raised Recall@10 from {num(ch['recall10'])} to {num(dep['recall10'])} and improved the top-10 recovery of "
        "cold-start positives by more than an order of magnitude.",
        "Availability awareness in candidate generation is the key to the efficiency of SARA. Standard reciprocal rank "
        "fusion implicitly treats a missing source as the worst rank and thereby systematically displaces candidates that "
        "lack semantic information; the fusion of SARA reduces exactly to the retrieval order when semantics are "
        "unavailable, so that poorly annotated positives such as ncRNAs are largely retained. At the same time, the "
        "semantic order moves cold-start positives with very low retrieval ranks into the candidate range, which reduces "
        f"the budget needed to cover half of the test positives by about {pc(1 - N['K50_r_aware'] / N['K50_rB'], 0)}. Model "
        "explanation further clarifies the decision logic of the ranking stage: the feature gain of the ranker comes mainly "
        f"from the retrieval channels ({pc(fam['V6 检索通道'], 0)}) and the fusion rank ({pc(fam['候选融合'], 0)}), with "
        f"semantic features accounting for about {pc(fam['语义'], 0)}; for cold-start positives, however, the semantic "
        f"family ({sgn(ca['语义'], 2)}) and the fusion rank ({sgn(ca['候选融合'], 2)}) provide the main positive "
        f"contributions, whereas the contribution of the retrieval channels is negative ({sgn(ca['V6 检索通道'], 2)}). In "
        "other words, SARA does not replace retrieval evidence with semantic evidence but lets semantic evidence take over "
        "where retrieval evidence is missing, which is precisely the aim of availability-aware design.",
        "### 4.2 Comparison with existing disease gene prioritization methods",
        "Network propagation methods [@kohler2008; @vanunu2010] and phenotype-similarity methods [@kohler2009; "
        "@smedley2015] start from known associations or phenotype annotations and perform well for diseases and genes with "
        "complete information, but their performance depends to a considerable extent on a few highly connected ‘hub’ "
        "genes [@gillis2012]. The stratified results of this study agree: the retrieval channels were stable for genes with "
        "rich reference associations but almost powerless for cold-start positives. Knowledge-graph completion and graph "
        "neural network methods [@wang2024kdgene; @frasca2020] likewise depend on the graph structure visible during "
        "training; in the family-disjoint setting, in which test diseases take no part in training, KDGene and the "
        f"SIGN-style model retrained on the same candidate lists reached Recall@10 of {num(comp['kdgene']['recall10'])} and "
        f"{num(comp['sign']['recall10'])}, far below SARA, which suggests that part of the advantage of such methods may "
        "rely on structural overlap between training and test diseases. Unlike these approaches, SARA uses the semantic "
        "representations of language and protein language models [@liu2021sapbert; @lin2023esm2] as an evidence source "
        "orthogonal to known associations and combines them with the retrieval channels in an availability-aware manner, "
        "extending the range of retrievable genes without sacrificing performance on candidates with complete information.",
        "The evaluation framework of SARA may also serve as a reference for disease gene retrieval research. Training, "
        "validation and test diseases were separated at the family level, all second-level features were generated "
        "out-of-fold in inner contexts and all configurations were selected on separate development families; three-layer "
        "evidence deletion with matched random controls tested the influence of shared literature directly; and controlled "
        "masking of functional annotations together with the withdrawal of association supervision turned the question of "
        "where semantic evidence comes from into a testable one. These designs allow the gains of SARA to be attributed to "
        "specific sources of information rather than to leakage during evaluation [@kaufman2012]. The stratified analyses "
        "also showed that the gain of SARA grows with the richness of functional annotations, which suggests that better "
        "functional prediction for poorly annotated genes (for example with protein language models [@lin2023esm2; "
        "@xu2023protst]) is the most direct way to extend the coverage of SARA.",
        "### 4.3 Limitations and future directions",
        "This study has several limitations. First, the positive associations come from the manually curated CTD database; "
        "unlabeled candidates are not necessarily negatives [@bekker2020], and highly ranked new candidates require "
        "experimental or literature validation. Second, the cross-validation folds and the external label set were used "
        "during the iterative development of the method, so the results constitute retrospective internal-validation "
        "evidence that still requires independent testing on prospective or entirely new data; in temporal extrapolation "
        "with resources rebuilt as of the cut-off date, SARA and retrieval-channel LightGBM performed comparably. Third, "
        "the semantic gains of SARA depend on functional-annotation coverage and offer limited help for genes without "
        "annotations. Fourth, the expression cohorts provide only correlational evidence. Future work will validate SARA "
        "on prospective data and on further external resources such as DisGeNET [@pinero2020] and will explore "
        "high-quality functional prediction to extend its coverage of poorly annotated genes.",
        "## 5 Conclusion",
        "This study presents SARA, a disease gene retrieval framework designed for uneven knowledge coverage. Through "
        "disease–function semantic retrieval, availability-aware candidate fusion and missingness-explicit learning to "
        "rank, SARA significantly improved disease gene retrieval and the retrievability of cold-start genes under strict "
        "family-disjoint evaluation, and it showed a consistent direction of benefit on external labels, in support tasks "
        "and in independent transcriptomic cohorts. Its interpretable and auditable design offers a general framework for "
        "exploiting heterogeneous and incomplete biomedical knowledge.",
    ])


def endmatter() -> str:
    return "\n\n".join([
        "## Data availability",
        "All data used in this study come from public resources: CTD, HPO, Gene Ontology and GOA human annotations, MeSH, "
        "the Human Disease Ontology, OMIM, HGNC, STRING v12, the Human Protein Atlas, MANE and UniProt, GTEx v8, GenCC, "
        "ClinVar and GEO (GSE13355, GSE30999, GSE87466 and GSE75214). The code of SARA, the frozen model configurations, "
        "the plotting data of every figure panel (CSV) and all evaluation results will be made publicly available at "
        "[code repository link] and [data archive DOI] upon acceptance; the versions and download dates of all resources "
        "are provided with the code.",
        "## Funding", "[Funding bodies and grant numbers]",
        "## Conflict of interest", "[The authors declare no conflict of interest, or list any conflicts of interest]",
        "## Author contributions",
        "[Author A: conceptualization, methodology, software, formal analysis, writing – original draft; Author B: …; "
        "corresponding author: supervision, writing – review and editing]",
        "## Acknowledgements",
        "[Acknowledgements] [If artificial intelligence tools were used in the research or writing, describe their use and "
        "scope here as required by the journal]",
        "## Author biographies",
        "[Author name] is a [position] at [institution]. [His/her/their] research interests include [research interests].",
    ])


# ============================================================================ tables
def table1(R, N) -> str:
    z = N["z"]
    M = _meta(N)
    au = z["audit"]
    l3 = au["layer3_evidence_blocks"]
    hc = z["hist"]
    x = N["x"]
    s1 = {r["method"]: r for r in z["support1"]}
    tx = json.loads((paths8.AUDIT / "text_segments_l1.json").read_text(encoding="utf-8"))
    mk = json.loads((paths8.MANIFESTS / "go_masks_v8.json").read_text(encoding="utf-8"))
    rows = [
        ["Gene background", i0(M["G"]), f"protein-coding {i0(M['protein'])}, non-coding RNA {i0(M['ncrna'])}, "
                                        f"pseudogenes {i0(M['pseudo'])}, other {i0(M['other'])}"],
        ["Positive associations / diseases with positives", f"{i0(M['n_rel'])} / {i0(M['dis_rel'])}",
         "CTD marker/mechanism, direct evidence"],
        ["Test diseases / associations / families", f"{i0(M['test'])} / {i0(M['test_rel'])} / {i0(M['test_fam'])}",
         "five outer folds, family-disjoint"],
        ["Development / external hold-out diseases", f"{i0(M['dev'])} / {i0(M['ext_heldout'])}",
         "development set used only for configuration choices; external hold-out never used for training"],
        ["GO records / genes with GO", f"{i0(M['go_rec'])} / {i0(M['has_go'])}", "GAF 2.2; NOT and ND removed"],
        ["Genes with sequence / expression", f"{i0(M['has_seq'])} / {i0(M['has_expr'])}",
         "representative protein (ESM-2 35M); HPA consensus tissue expression"],
        ["Genes with gene–phenotype annotations / physical interactions", f"{i0(M['has_ghpo'])} / {i0(M['has_ppi'])}",
         "source-excluded HPO; STRING v12 (≥700)"],
        ["Long-text diseases encoded in segments", f"{i0(tx['n_truncated_v71'])} ({pc(tx['share_truncated_v71'])})",
         f"{i0(tx['segments_total'])} segments in total, ≤126 word pieces each"],
        ["GO mask sets (development / test)", f"{i0(mk['G_mask_dev'])} / {i0(mk['G_mask_test'])}",
         "stratified hashing, no labels read"],
        ["Evidence blocks (blocks / associations / publications)",
         f"{i0(l3['blocks_chosen'])} / {i0(l3['relations_held'])} / {i0(l3['pmids_held'])}",
         f"giant component of {i0(l3['giant_component_relations'])} associations not split"],
        ["External labels (diseases / pairs)", f"{i0(z['ext']['fields']['diseases'])} / {i0(M['ext_union'].get('n_pairs', 0))}",
         "GenCC and ClinVar"],
        ["Temporal extrapolation (new associations / new diseases)",
         f"{i0(hc['x2_deployed_vs_channel_lgb_v6']['n'])} diseases, {i0(x['hp']['x2']['pairs'])} pairs / "
         f"{i0(hc['x1_deployed_vs_channel_lgb_v6']['n'])} diseases, {i0(x['hp']['x1']['pairs'])} pairs", "cut-off 2023-12-31"],
        ["Support tasks", f"533 diseases ($k\\in\\{{0,1,3,5\\}}$) / {i0(s1['joint_v6']['n_diseases'])} diseases ($k=1$)",
         "known support genes revealed"],
        ["Expression cohorts", "4", "GSE13355, GSE30999, GSE87466, GSE75214"],
    ]
    return "Table 1. Data sources and evaluation sets\n\n" + table_md(["Item", "Number", "Description"], rows)


def table2(R, N) -> str:
    z = N["z"]
    rows = []
    for m in TABLE2_ORDER:
        r = z["table"][m]
        name = "**SARA**" if m == "deployed" else method(m, table=True)
        rows.append([name, num(r["recall10"]), num(r["recall20"]), num(r["recall50"]), num(r["hit10"]),
                     num(r["coverage500"]), num(r["U10"]), num(r["cold_recall10"])])
    t = table_md(["Method", "R@10", "R@20", "R@50", "Hit@10", "Coverage@500", "U@10", "$R^c$@10"], rows)
    return (f"Table 2. Retrieval performance of all methods on the test set ({i0(z['table']['deployed']['n_diseases'])} "
            "test diseases, mean of three seeds)\n\n" + t +
            "\n\nNote: Except where ‘retrieval top 500’ is indicated, all ranking models were evaluated on the SARA "
            "candidates; stand-alone semantic rankings take the full-background top 500. $R^c$@10 is defined only on the "
            f"{i0(N['cold'][0]['n_cold_diseases'])} diseases with cold-start positives. KDGene and the SIGN-style graph "
            "model retrained on the same candidate lists (seed 42) reached Recall@10 of "
            f"{num(z['comp']['kdgene']['recall10'])} and {num(z['comp']['sign']['recall10'])} (Supplementary Table S3).")


def table3(R, N) -> str:
    z = N["z"]
    b = {(r["table"], r["method"]): r for r in z["budget"]["table"]}
    v = z["var"]
    miss = pd.DataFrame(N["miss"])

    def nc(li):
        return pc(_row(miss, group="非编码 RNA", list=li).get("pool_share"))
    ch6, a6 = b[("1_fixed_v6_500", "仅通道 LightGBM")], b[("1_fixed_v6_500", "候选 A（部署规则）")]
    a5, a9 = b[("2_shared_new_500", "候选 A（部署）")], b[("4_bridge_950", "候选 A（同 950 预算）")]
    c9 = b[("4_bridge_950", "仅通道 LightGBM（同 950 预算）")]
    pr, eo, er = v["plain_rrf"]["deployed"], v["expr_off"]["deployed"], v["expr_random"]["deployed"]

    def d(e, mark):
        return ci(e, label=False) + mark
    rows = [
        ["Retrieval top 500", "Retrieval-channel LightGBM", num(ch6["recall10"]), pc(ch6["coverage"]),
         nc("V6 500（仅通道）"), "—"],
        ["Retrieval top 500", "SARA ranker", num(a6["recall10"]), pc(a6["coverage"]), nc("V6 500（仅通道）"),
         d(z["budget"]["v6pool_A_vs_channel"], "†")],
        ["SARA candidates", "SARA", num(a5["recall10"]), pc(a5["coverage"]), nc("缺失感知严格 500"),
         d(z["budget"]["strict_A_vs_v6pool_A"], "†")],
        ["Standard RRF candidates", "SARA ranker", num(pr["recall10"]), pc(pr["coverage500"]), nc("普通 RRF 严格 500"),
         d(pr["delta_vs_main"], "‡")],
        ["Extended candidates (~950)", "SARA ranker", num(a9["recall10"]), pc(a9["coverage"]), "—",
         d(z["budget"]["bridge_A_vs_strict_A"], "‡")],
        ["Extended candidates (~950)", "Base ranker", num(c9["recall10"]), pc(c9["coverage"]), "—",
         d(z["budget"]["bridge_A_vs_B"], "§")],
        ["SARA candidates", "SARA, expression features off", num(eo["recall10"]), pc(eo["coverage500"]), "—",
         d(eo["delta_vs_main"], "‡")],
        ["SARA candidates", "SARA, random tissue expression", num(er["recall10"]), pc(er["coverage500"]), "—",
         d(er["delta_vs_main"], "‡")],
    ]
    return ("Table 3. Ablation of candidate budget, fusion type and expression features (seed 42)\n\n" +
            table_md(["Candidate list", "Ranking model", "R@10", "Coverage", "ncRNA listed", "ΔR@10 (95% CI)"], rows) +
            "\n\nNote: Coverage is the share of test positives in the candidate list; ‘ncRNA listed’ is the share of "
            "non-coding RNA positives entering the candidates (association level). ΔR@10: † minus the row above; ‡ minus "
            "SARA on the SARA candidates; § SARA ranker minus base ranker on the same extended candidates. All lists hold "
            "500 genes per disease except the extended candidates, the union of the retrieval top 500 and the semantic top "
            f"500 (on average {i0(a9['pool_size'])} genes); every ranking model was retrained out-of-fold on its list.")


# ============================================================================ figure legends
FIG_TITLES = {
    "1": "Distribution of disease–gene knowledge and availability of information layers",
    "2": "The SARA framework and the nested cross-validation design",
    "3": "Retrieval performance of SARA compared with baseline methods",
    "4": "Candidate budget, fusion type and rank flows",
    "5": "Retrieval performance across levels of knowledge coverage",
    "6": "Feature contributions of the SARA ranker and TreeSHAP explanations",
    "7": "Effect of removing functional annotations and withdrawing association supervision on masked genes",
    "8": "Sources of GO evidence and sensitivity to evidence deletion",
    "9": "External labels, temporal extrapolation and application tasks",
    "10": "Top-ranked genes and gene interaction networks of representative diseases",
}
CI_NOTE = "Intervals are bootstrap 95% CIs from resampling disease families."


def legends(R, N) -> list[tuple[str, str, str]]:
    M = _meta(N)
    L = {
        "1": ("**a** Lorenz curves: cumulative share of known associations with genes (or diseases) ordered by increasing "
              "association count; the shading marks the area between the gene curve and the line of equality, and the dot "
              "marks the share held by the top 10% of genes. **b** Complementary cumulative distributions $P(X\\geq k)$ of "
              "three counts (log–log axes; only objects with a count of at least 1). **c** Distribution of known "
              "association counts within groups of genes defined by GO annotation count (column-wise shares; cells of at "
              "least 10% are labeled; top, number of genes per group); $\\rho$ is the Spearman correlation over all genes. "
              "**d** UpSet plot of the six information layers: top, gene counts of the 14 largest intersections (dark, "
              "protein-coding); bottom, dot matrix marking the layers of each intersection; left, the size of each layer. **e** Completeness "
              "of disease descriptions by known association count (numbers of diseases in parentheses). **f** Chord "
              "diagram of the associated genes shared among the 12 MeSH categories with the most associated genes: arc "
              "length is proportional to the number of associated genes in a category and ribbon width to the number of "
              "genes shared by two categories; arc color gives the median number of GO gene–term pairs of the category's "
              "associated genes, and blue ribbons mark the six pairs with the highest Jaccard coefficients. **g** Nine "
              "knowledge-coverage indicators of each category with at least 40 associated diseases — disease side: number "
              "of diseases, median associations per disease, shares with a definition and with HPO, median text tokens; "
              "gene side (associated genes of the category): median GO gene–term pairs, median known associations, shares "
              "with physical interactions and without GO. Colors are column-wise $z$ scores (red low, blue high) and cells "
              "show raw values; the dendrograms on the left and top come from average-linkage hierarchical clustering. "
              "In total, "
              f"{i0(M['G'])} genes and {i0(M['n_dis'])} diseases."),
        "2": ("**a** The SARA pipeline: multi-source retrieval channels give a retrieval order $r_B$ over all background "
              "genes, and the disease–function semantic model gives a semantic score $s_F$ and a semantic rank $r_S$; "
              "availability-aware RRF generates exactly 500 candidates per disease, and the SARA ranker re-ranks them with "
              "49 out-of-fold features (the box lists the mean gain share of each feature family); dashed arrows mark "
              "features that enter the ranker directly. **b** The five outer folds: dark, test fold; mid-gray, validation "
              "fold; light, training folds, which are split by family into three inner blocks $I_{jb}$; right, the "
              "development, external and temporal sets kept apart from the five folds. **c** The availability-aware score "
              "$R$ as a function of $r_B$ and $r_S$ at $w=0.5$ (log color scale); the black line is the median boundary of "
              "the 500th candidate score over test diseases and white lines are quartile boundaries; the bar below shows "
              "the score when semantics are unavailable ($m=0$), with vertical lines at the listing cut-offs of standard "
              "and availability-aware RRF. **d** The status of the information items under the three kinds of ‘cold’ "
              "(filled, retained; open, withdrawn or missing). **e** UpSet plot of the availability of the seven retrieval "
              "sources over the test diseases (five folds combined): top, number of diseases with each availability "
              "pattern; middle, the available sources of each pattern; left, number of diseases for which each source is "
              "available; bottom, per-disease Recall@10 difference between SARA and retrieval-channel LightGBM within each "
              "pattern (three-seed mean; dots and bars, estimates and 95% CIs; blue, lower bound above zero; gray, CI "
              "including zero). " + CI_NOTE),
        "3": ("**a** Distributions of the primary differences: half-violins show the distributions of the differences over "
              "10,000 family bootstraps, dots and horizontal lines the estimates and 95% CIs; on the right, the values and "
              "whether the CI includes zero. **b** Cold-start recovery: filled symbols, $R^c$@10 (defined only for "
              "diseases with cold-start positives); open symbols, $R^{c\\to all}$@10 (cold-start positives in the top 10 "
              "as a share of all test positives). **c** Recall@$K$ curves ($K$ from 1 to 100, log axis); bands are "
              "pointwise 95% bands from 1,000 family bootstraps. **d** Differences between SARA and retrieval-channel "
              "LightGBM in the 15 outer-fold × seed combinations (shapes distinguish seeds; black diamonds, pooled "
              "estimates over three seeds). **e** All metrics (colored within columns; the colored squares on the right "
              "give the fixed color of each method). **f** Shares of per-disease wins and losses (comparing the three-seed "
              "mean Recall@10 per disease); numbers of diseases in parentheses. **g** Raincloud plots of the best positive "
              "rank per disease (seed 42; top, kernel density; middle, box plot; bottom, sampled per-disease points; only "
              "diseases with a listed positive; log axis); right, share of diseases without a listed positive. **h** Chord "
              "diagram of the overlap of the top-10 hits (disease–gene pairs, seed 42) of five methods: arc length is "
              "proportional to the number of top-10 hits of a method and ribbon width to the number of hits shared by two "
              "methods. **i** Radar chart of seven metrics (each axis divided by the maximum of the four methods; MRR is "
              "the mean reciprocal rank of positives within a disease). In total, "
              f"{i0(M['test'])} test diseases and {i0(M['test_fam'])} families. " + CI_NOTE),
        "4": ("**a** Share of test positives in the top $K$ (macro-average over diseases; log-scaled $K$ axis) for the "
              "retrieval order $r_B$, the semantic order $r_S$ (semantically unavailable genes counted as not listed), "
              "standard RRF and availability-aware RRF (with the $w_j$ of each outer fold); bands are 95% bands, and "
              "vertical lines mark $K=500$ and $K\\approx 950$. **b** Budget attribution: Recall@10 of the same ranking "
              "model on the retrieval top 500, the SARA candidates and the extended candidates (~950) (seed 42; bars, 95% "
              "CIs), with the three attribution differences at the bottom left. **c** Share of positives lacking "
              "information that enter the candidates under three lists (association-level micro-average; numbers of "
              "associations in parentheses; error bars, family-clustered bootstrap intervals). **d** Flows of "
              "per-association rank bins (left, retrieval-channel LightGBM; right, SARA); blue, rank up; orange, rank down; "
              "gray, same bin. **e** Two-dimensional distribution of the ranks of the same positives under the two methods "
              "(log color scale). **f** Joint distribution of semantically available test positives over the retrieval "
              "rank $r_B$ and the semantic rank $r_S$ (seed 42; hexagonal bins, log color scale; integer ranks $r$ are "
              "jittered uniformly within $\\left[r,r+1\\right)$ before taking logarithms, for display only): dashed lines "
              "are two-dimensional kernel density contours of cold-start positives (50th, 80th and 95th percentiles of the "
              "grid densities), the black line is the median listing boundary of availability-aware RRF at $w=0.5$, and "
              "orange and purple lines mark $r_B=500$ and $r_S=500$; top and right, marginal distributions (dark, "
              "cold-start positives); corners, share of positives in each quadrant. **g** Fate of each class of positives "
              "when the retrieval top 500 is replaced by the SARA candidates (seed 42, association level). " + CI_NOTE),
        "5": ("**a** Association-level top-10 recovery stratified by the number of reference associations of the gene in the "
              "evaluation context (three-seed mean; ‘cold’, no reference edge and no gene–phenotype annotation; ‘0\\*’, no "
              "reference edge but a gene–phenotype annotation; the numbers at the top are associations per seed). **b** "
              "Stratified by the number of GO gene–term pairs of the gene. **c** Per-disease Recall@10 difference (SARA "
              "minus retrieval-channel LightGBM) stratified by disease text-length quartile and by the presence of HPO "
              "(cells show the difference and the number of diseases; cells with fewer than 20 diseases are not colored). "
              "**d** MeSH categories: left, Recall@10 of four methods (common color scale); middle, differences of SARA "
              "from retrieval-channel LightGBM (dots) and from multimodal direct fusion (diamonds); right, number of test "
              "diseases. **e** Ridgeline plots of the log improvement of the best positive rank per disease, "
              "$\\log_{10}r_{\\mathrm{C}}^{*}-\\log_{10}r_{\\mathrm{S}}^{*}$, by MeSH category ($r_{\\mathrm{C}}^{*}$ and "
              "$r_{\\mathrm{S}}^{*}$ are the best positive ranks of retrieval-channel LightGBM and SARA; seed 42; a positive "
              "not listed counts as 501; only diseases with a listed positive under at least one method): fill color is "
              "the mean improvement of the category (red negative, blue positive), and the right column gives the shares "
              "of diseases with an improved (↑) or worsened (↓) best rank and the number of diseases. " + CI_NOTE),
        "6": ("**a** Gain share of the features of the SARA ranker (mean ± s.d. over 15 models), colored by feature family; "
              "right, the total of each family. **b** Mean signed SHAP value of each feature family by candidate class "
              "(numbers of candidates in parentheses). **c** TreeSHAP beeswarm plot of the SARA ranker of outer fold 0 and "
              "seed 42 on test candidates (all held-out positives and 40,000 sampled unlabeled candidates; the 14 features "
              "with the largest mean absolute SHAP values; point color gives the relative position of the feature value "
              "between its 2nd and 98th percentiles, gray marks missing values). **d** SHAP dependence of the fusion rank "
              "(availability-aware RRF rank, log axis); **e** SHAP dependence of the semantic score $s_F$: point color "
              "gives the retrieval rank $r_B$ (diverging scale centered at 500), open circles are 700 sampled held-out "
              "positives, and the black line and gray band are the binned mean and the 10th–90th percentile range. **f** "
              "Ablation of expression features: difference in the Recall@10 of SARA from the full model after turning off "
              "all expression features or replacing them with random tissue expression (seed 42; horizontal bars, 95% "
              "CIs). Panels b–e explain the model of a single context."),
        "7": ("**a** Cumulative distributions of the number of GO terms and of the physical-interaction degree for unmasked "
              "genes, the development mask set (10%) and the test mask set (20%) (genes with degree 0 are drawn at 0.5). "
              "**b** Shares of the positives of test-masked genes that enter the candidates (light) and the top 10 (dark) "
              "for the full model, after GO removal, after withdrawal of association supervision and after both (seed 42; "
              "horizontal bars, 95% CIs); right, change in the Recall@10 over all diseases relative to the full model. "
              "**c** Ridgeline plots of the ranks of masked-gene positives among the SARA candidates (seed 42, all outer "
              "folds; listed positives only; log axis; kernel densities with reflection at the bounds; vertical lines, "
              "medians); right, the listed share and the top-10 share. **d** Flows of the rank classes of the same "
              "positives between the full model, GO removal and ‘supervision withdrawn + GO removed’ (top 10, 11–50, "
              "51–500 and not listed; ribbon width is proportional to the number of positives). " + CI_NOTE),
        "8": ("**a** Number of production GO records by evidence code (dark, with PMID; light, without PMID), with the code "
              "class in parentheses and the share without PMID on the right. **b** Left, size distribution of the connected "
              "components of the PMID–association bipartite graph (log–log axes; red dot, giant component); right, share "
              "of PMID-linked associations covered by blocks accumulated in decreasing order of size. **c** Deletion shares "
              "of each condition over five denominators (log color scale): citations, annotation rows, unique gene–term "
              "pairs, affected genes and affected query associations; * marks the sensitivity conditions that use the "
              "literature of evaluation associations. **d** Change in the advantage of SARA over retrieval-channel "
              "LightGBM between reduced and full data (seed 42, full-pipeline retraining; squares, matched random "
              "controls; gray, re-scoring sensitivity analyses that replace only the inference inputs). **e** The same "
              "change broken down by MeSH category (values ×100; categories with at least 60 test diseases): rows are the "
              "11 deletion conditions and columns are clustered by average linkage (top, dendrogram and number of diseases "
              "per category), with all diseases in the right column; blue means that the advantage of SARA grows after "
              "the deletion and red that it shrinks. " + CI_NOTE),
        "9": ("**a** Recall@10 differences between SARA and three comparators on external labels (GenCC and ClinVar), in "
              "rows for the union, GenCC, ClinVar and the literature-independent subset (numbers of diseases in "
              "parentheses). **b** Composition by source of three eligibility fields of the external associations (whether "
              "the disease appeared in training, whether the same association appears in neighboring diseases and whether "
              "the association shares source literature with training associations). **c** Temporal extrapolation "
              "(cut-off 2023-12-31): Recall@10 of each method on the new-association and new-disease tasks (bars, means; "
              "vertical lines, 95% CIs) and the difference between SARA and retrieval-channel LightGBM. **d** Support task "
              "(533 diseases, $k$ support genes revealed; bands, 95% bands). **e** Enrichment of the frozen gene sets in "
              "four expression cohorts (color, probability that the $\\left|t\\right|$ of a selected gene exceeds that of "
              "matched background genes, 0.5 meaning no enrichment; circle area is proportional to $-\\log_{10}P$; black "
              "rings mark $P<0.05$). **f** Rank of every test association of the temporal tasks (dots, new-association "
              "task; triangles, new-disease task; seed 42) under retrieval-channel LightGBM (retrieval top 500) and SARA "
              "(SARA candidates), log axes, with unlisted associations drawn outside the separating bands; below the "
              "diagonal (blue background) SARA ranks higher, above it (orange background) retrieval-channel LightGBM ranks "
              "higher. **g** Per-disease Recall@10 in the support task ($k=1$ and $k=5$ support genes revealed; $k=1$ "
              "averaged over 10 draws, $k=5$ a single draw), square-root axes; blue dots, SARA semantics + RWR higher; "
              "orange dots, baseline semantics + RWR higher; gray dots, equal; gray background, point density. " + CI_NOTE),
        "10": ("**a** Two rows per case: top, SARA; bottom, retrieval-channel LightGBM; top 10 genes (seed 42, gene "
               "symbols); colored cells are held-out true associations (CTD marker/mechanism), a black frame and ° mark "
               "cold-start positives (reference-path cold in the evaluation context), and light gray cells are genes not "
               "annotated as associated. Cases were selected by a rule fixed before plotting (see text). **b** Top-10 gene "
               "networks of three cases: nodes are the union of the top 10 of both methods (left arc, SARA top 10 only; "
               "right arc, retrieval-channel LightGBM top 10 only; middle column, both), numbers inside nodes are ranks in "
               "the method's top 10, node area is proportional to $\\log\\left(1+n_{\\mathrm{GO}}\\right)$ "
               "($n_{\\mathrm{GO}}$, the number of GO gene–term pairs) and black rings mark held-out positives (° cold-start "
               "positives); edges are STRING v12 physical interactions (dark gray) and functional associations (light "
               "gray) and gene pairs whose GO term sets have a Jaccard coefficient of at least 0.2 (purple dashed)."),
    }
    return [(n, stem, f"**Figure {n}. {FIG_TITLES[n]}.** " + L[n]) for n, stem, _ in FIGS10]


# ============================================================================ assembly
def compose(R=None, N=None) -> tuple[str, dict]:
    from .refs_en import resolve
    if R is None:
        R, N = numbers9()
    TRACE.clear()
    body = [front(R, N), intro(R, N), methods(R, N), results(R, N), discussion(R, N), endmatter(),
            "## References", "<<REFERENCES>>", "## Tables", table1(R, N), table2(R, N), table3(R, N), "## Figure legends"]
    for n, stem, leg in legends(R, N):
        body += [f"<!-- figure {stem} -->", leg]
    md = "\n\n".join(p.strip("\n") for p in body) + "\n"
    md = md.replace("\\qquad ", QQUAD)
    md, eqs = number_equations(md)
    md, refs, order, recs = resolve(md)
    md = typography(md.replace("<<REFERENCES>>", "\n".join(refs)))
    info = {"stem": STEM, "title": TITLE, "ref_order": order, "n_refs": len(refs), "equations": eqs,
            "figures": [(n, stem) for n, stem, _ in FIGS10], "trace": {k: v for k, v in TRACE.items()}}
    return md, info


def write() -> dict:
    from .refs_en import CACHE, record, ris
    R, N = numbers9()
    md, info = compose(R, N)
    EN.mkdir(parents=True, exist_ok=True)
    (EN / f"{STEM}.md").write_text(md, encoding="utf-8")
    (ROOT / "build" / "en").mkdir(parents=True, exist_ok=True)
    (ROOT / "build" / "en" / "manuscript_en_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=1),
                                                                    encoding="utf-8")
    cache = json.loads(CACHE.read_text(encoding="utf-8"))
    recs = [record(k, cache) for k, _ in sorted(info["ref_order"].items(), key=lambda t: t[1])]
    (EN / f"SARA_references_{DATE}.ris").write_text(ris(recs), encoding="utf-8")
    words = len([w for w in md.split("## References")[0].split() if w.strip()])
    return {"md": str(EN / f"{STEM}.md"), "chars": len(md), "words_to_references": words, "refs": info["n_refs"],
            "equations": len(info["equations"])}


if __name__ == "__main__":
    print(json.dumps(write(), ensure_ascii=False, indent=1))

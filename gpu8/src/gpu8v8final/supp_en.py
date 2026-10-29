"""SARA supplementary material in English: Supplementary Notes S1-S6, Tables S1-S10 and the legends of Figures S1-S8.

The English counterpart of supp10 (= supp9 + round-10 edits).  Numbers are read from the same saved results through
the same helpers of supp9 and formatted by ``fmt_en`` (traced); citations are numbered separately for the supplement.
"""
from __future__ import annotations

import json

import pandas as pd

from . import paths8
from . import supp9 as S
from .fmt_en import TRACE, ci, cin, i0, num, pc, sci, sgn
from .manuscript10 import number_equations
from .manuscript9 import PUB, _row, table_md
from .manuscript_en import DATE, EN, TITLE, typography
from .numbers9 import ROOT, numbers9
from .terms_en import (CASE, CONTRAST, E4, E5, EXPR_SET, FAMILY, FEATURE, HIST, LATER, ROLE, SEMANTIC, SUPPORT, cap,
                       method, mesh_table)

SUPP_STEM = f"SARA_supplementary_{DATE}"


def _cfg(name: str) -> str:
    a, b = name.replace("A_nl", "").split("_mc")
    return f"{a} / {b}"


# ============================================================================ notes
def note_s1(R, N) -> str:
    tx = S.J(paths8.AUDIT / "text_segments_l1.json")
    lr = R["acceptance"]["1_label_removal"]
    return "\n\n".join([
        "## Supplementary Note S1. Data processing and information boundaries",
        "The sources of the gene background, the diseases and the positive associations are given in Section 2.1 and "
        "Table 1 of the main text; three further processing details are described here.",
        f"**Disease text.** Disease text consists of the name, the definition and up to eight synonyms; before encoding, "
        f"gene symbols are masked with a pre-fixed dictionary of HGNC-approved symbols. After segmentation with priority "
        f"for sentence boundaries, the {i0(tx['n_diseases'])} diseases yielded {i0(tx['segments_total'])} segments, with at "
        f"most {tx['max_segments']} segments per disease; the full texts had a median of "
        f"{num(tx['tokens_full_median'], 0)} word pieces and a 95th percentile of {num(tx['tokens_full_p95'], 0)}, and no "
        f"segment was truncated by the tokenizer ({i0(tx['segment_tokenizer_truncations'])} truncations). For diseases "
        "with a single segment, the vector is identical, element by element, to the unsegmented encoding.",
        "**Phenotype annotations.** Disease HPO annotations are attached through strict equivalence mappings, keeping only "
        "terms of the phenotypic-abnormality branch without a NOT qualifier and at a depth greater than 2. Every row of "
        "the gene–phenotype annotations carries its source disease and is aggregated to genes only after all rows whose "
        "source disease is a held-out disease of the current context, or belongs to its strict equivalence group, have "
        "been removed; rows whose source cannot be mapped are excluded from the main analysis.",
        "**Information boundaries.** The reference set of each context excludes the strict equivalence group of every "
        "held-out disease, and all inputs of the retrieval channels, the semantic model, the candidate lists and the "
        "ranker are generated according to the role of each disease (training, validation, test, development or external "
        "hold-out) without reading test labels. When the inputs were regenerated after the test labels of "
        f"{i0(lr['n_checked_diseases'])} test diseases of fold {lr['fold']} had been removed, the retrieval orders were "
        "identical element by element, the candidate lists were identical and the maximum difference in SARA scores was "
        f"{sci(lr['deployed_score_maxabs'])}.",
    ])


def note_s2(R, N) -> str:
    rv = S.J(paths8.GPU8_ROOT / "manifests_v6" / "retriever_v6_frozen.json")
    sv = S.J(paths8.GPU8_ROOT / "manifests_v6" / "scorer_v6_frozen.json")
    w, fw = rv["weights"], sv["weights"]
    tc = S._teacher(0)["config"]["train"]
    ctl = S.J(paths8.MANIFESTS / "dev_teacher_controls_v8.json")["per_context"]
    n_bg = sum(1 for v in ctl.values() if v["controls"] == "background")
    gap = max(abs(S._sel(j)["w"]["coverage"][f"aware_w{x}"] - S._sel(j)["w"]["coverage"][f"plain_w{x}"])
              for j in range(5) for x in (0.3, 0.5, 0.7, 0.9))
    return "\n\n".join([
        "## Supplementary Note S2. Implementation of the retrieval channels, the semantic model and candidate generation",
        "**Retrieval channels.** The ranks of the seven sources over the whole background are combined by reciprocal rank "
        "fusion ($K_0=60$), with unavailable sources masked and their weights redistributed. The fusion type and the "
        f"weights were selected on the development set only ({i0(sv['n_dev_disease_contexts'])} ‘disease × outer "
        f"context’ pairs): {i0(rv['n_grid_configs'])} configurations on a simplex grid with a step of 0.25 and two fusion "
        f"types were evaluated first, followed by {i0(rv['n_refine_configs'])} neighborhood refinements with a step of "
        "0.05, with the macro-averaged U@10 at a candidate budget of 500 as the objective. The frozen weights are "
        f"hierarchy {num(w['hier'], 2)}, ontology {num(w['onto'], 2)}, text {num(w['text'], 2)}, gene–phenotype "
        f"{num(w['gene_hpo'], 2)} and phenotype propagation {num(w['rwr_pheno'], 2)}. The phenotype-transfer and "
        "compatibility sources have a fusion weight of 0 — the signal of the former enters through the phenotype "
        "propagation that it seeds — and the scores, ranks and availability of both still enter the ranker as features. "
        "The weights of the fixed-weight fusion comparator were likewise selected on the development set (hierarchy "
        f"{num(fw['hier'], 2)}, ontology {num(fw['onto'], 2)}, text {num(fw['text'], 2)} and gene–phenotype "
        f"{num(fw['gene_hpo'], 2)}).",
        "**Semantic model.** The HPO representation of a disease is the annotation-frequency-weighted mean of the SapBERT "
        "vectors of its HPO terms (missing frequencies set to 1), and no phenotype is imputed for diseases without HPO "
        "annotations. The GO functional anchors are the GO terms annotated to 5–500 background genes (transitive closure "
        "over is_a and part_of, with the three root terms removed), and positive anchors of the same gene are never used "
        f"as negatives for one another. Training starts with {tc['anchor_steps']} anchor-update steps and continues with "
        f"joint training: AdamW (learning rate {tc['lr']}, weight decay {tc['weight_decay']}), {tc['batch']} diseases and "
        f"{tc['anchor_batch']} anchor genes per batch, gradient-norm clipping at {tc['grad_clip']}, at most "
        f"{tc['max_epochs']} epochs and a patience of {tc['patience']}. For each outer context, the unlabeled control set "
        "was selected between ‘reference-associated genes’ and ‘full background’ by the $s_F$ Recall@10 on the "
        f"development set; {'all five' if n_bg == 5 else f'{n_bg} of the five'} outer contexts selected the full "
        "background (Supplementary Table S1, Supplementary Figure S1).",
        "**Candidate generation.** The RRF weight $w$ of each outer context was selected by the positive coverage@500 on "
        "the development set (Supplementary Figure S2a). Over the four candidate weights, the development coverage of "
        f"standard and availability-aware RRF differed by at most {num(gap, 4)}; the two differ mainly for positives that "
        "lack semantic information (Figure 4c of the main text).",
    ])


def note_s3(R, N) -> str:
    cf = [_cfg(S._sel(j)["ab"]["best_A"]) for j in range(5)]
    fi = pd.read_parquet(PUB / "feature_importance.parquet")
    nmod = fi.groupby(["seed", "fold"]).ngroups
    return "\n\n".join([
        "## Supplementary Note S3. Features and configuration selection of the SARA ranker",
        f"Supplementary Table S2 lists the 49 features of the SARA ranker with their mean gain shares over the {nmod} "
        "models (5 outer folds × 3 seeds). The configurations selected for the five outer contexts (number of leaves / "
        "minimum samples per leaf) were, in order, " + ", ".join(cf[:-1]) + f" and {cf[-1]} (Supplementary Table S1, "
        "Supplementary Figure S2b). The ranker is trained on the out-of-fold candidates and features of the three inner "
        "blocks and validated on the candidates of the outer $F_{j+1}$; missing semantic and expression features enter "
        "as NaN, and gene type and modality availability enter as explicit indicator features, so that the model can "
        "distinguish a low score from unavailable information.",
    ])


def note_s4(R, N) -> str:
    con = R["e2"]["contrasts"]
    sels = [S._sel(j) for j in range(5)]
    bc = [max(s["summary"]["corrector"], key=lambda r: r["dev_recall10"])["dev_recall10"] for s in sels]
    ba = [s["ab"]["best_A_dev_recall10"] for s in sels]
    gaps = [a - c for a, c in zip(ba, bc)]
    assert all(g > 0.005 for g in gaps), "a corrector fell within the development margin"
    st = S.J(PUB / "dev_misc.json")["student"]["per_context"]
    gains, cos_t, cos_p = [], [], []
    for v in st.values():
        for r in v["rows"]:
            if r["config"] == "none":
                continue
            gains.append(r["gain"])
            cos_t.append(r["gene_level_masked"]["mean_cos_true"])
            cos_p.append(r["gene_level_masked"]["mean_pairwise_cos"])
    assert max(gains) < 0, "an imputation configuration passed the development check"
    npar = {}
    for p in sorted(paths8.RUNS.glob("V8S__devmask__f0_outer__*__s42")):
        kind = p.name.split("__")[3].split("_")[0]
        npar[kind] = S.J(p / "result.json")["fit"]["n_params"]
    v = R["variants"]
    none_pool = v["e5_none"]["deployed"]["mask_test_pool_micro"]
    others = [v[t]["deployed"]["mask_test_pool_micro"] for t, _ in S.E5_PUB[1:7]]
    assert max(others) <= none_pool + 1e-12
    return "\n\n".join([
        "## Supplementary Note S4. Two alternative designs not included in the final model",
        "Before the final design of SARA was fixed, two further designs aimed at improving the recovery of cold-start "
        "genes were examined. Both were trained and selected on the same candidate lists under the same out-of-fold rules "
        "as SARA and were compared with the SARA ranker on the development set; because neither yielded a gain, neither "
        "entered the final model. Their design and results are reported here in full (Supplementary Figure S8, "
        "Supplementary Table S8).",
        "**Gated cold-balance corrector.** This design adds a residual gated by semantic availability to the base ranker "
        "$B$:",
        "$$s(d,g)=B(d,g)+a(d,g)\\,r_\\theta\\left(x_{dg}\\right)$$ {#eq:s1}",
        "where $a$ is the semantic availability gate and $x_{dg}$ comprises the same 49 features as used by the SARA "
        "ranker. The residual is trained with a positive–unlabeled pairwise loss [@elkan2008; @bekker2020], cold-start "
        "positives are weighted by $1+\\beta c_g$ ($c_g$ being the cold-start indicator), and the residual carries a "
        "quadratic penalty with coefficient $\\lambda$; the grid was $\\beta\\in\\{0,1,2\\}$ × "
        "$\\lambda\\in\\{0.03,0.1\\}$, implemented in LightGBM with a custom objective (15 leaves, at most 200 boosting "
        "rounds, with the number of rounds chosen by the Recall@10 on $F_{j+1}$). When $a=0$, $s\\equiv B$, so the "
        "design falls back exactly to the base ranker when semantics are unavailable. The selection rule was fixed in "
        "advance: a corrector would be considered only if its development Recall@10 was at least that of the best SARA "
        "ranker configuration minus 0.005. In all five outer contexts, the development Recall@10 of the best corrector "
        f"({num(min(bc))}–{num(max(bc))}) was more than {num(min(gaps))} below that of the best SARA ranker configuration "
        f"({num(min(ba))}–{num(max(ba))}) (Supplementary Figure S8a–b, Supplementary Table S1). The test-set comparison "
        "agreed: for $\\beta$ = 0, 1 and 2, the Recall@10 differences between the corrector and SARA were "
        f"{ci(con['corrector_beta0']['recall10_vs_deployed'])}, {ci(con['corrector_beta1']['recall10_vs_deployed'])} and "
        f"{ci(con['corrector_beta2']['recall10_vs_deployed'])}, respectively, whereas the cold-start recovery $R^c$@10 "
        f"was comparable to that of SARA ({cin(con['corrector_beta0']['cold_vs_deployed'])}; $\\beta=0$; "
        "Supplementary Figure S8c).",
        "**Sequence imputation model.** This design takes frozen ESM-2 35M sequence representations as input and "
        "regresses the functional representation $z^F_g$ of the semantic model of the same context (cosine distillation "
        f"[@hinton2015]) with a two-layer perceptron (480→256→128, GELU; {i0(npar['mlp'])} parameters) or a linear map "
        f"({i0(npar['linear'])} parameters); imputation is applied only to genes without observed GO annotations. "
        "Adoption required three conditions on the development mask set: a higher share of masked-gene positives entering "
        "the candidates, the disappearance of this gain when sequence pairings are shuffled within length strata, and a "
        "decrease in overall development coverage of at most 0.005. The changes of all four configurations in the five "
        f"outer contexts were negative ({sgn(min(gains))} to {sgn(max(gains))}; Supplementary Figure S8d), although the "
        f"mean cosine between imputed and true functional vectors ({num(min(cos_t), 2)}–{num(max(cos_t), 2)}) was clearly "
        f"higher than the mean pairwise cosine ({num(min(cos_p), 2)}–{num(max(cos_p), 2)}; Supplementary Figure S8e); the "
        "imputation model was therefore not adopted. On the test mask set, none of the six imputation variants raised "
        f"the share of masked-gene positives entering the candidates above the {pc(none_pool)} obtained without imputation "
        f"({pc(min(others))}–{pc(max(others))}; Supplementary Figure S8f–g, Supplementary Table S8).",
        "These two sets of results delimit the design of SARA: with the current data, sequence representations cannot yet "
        "replace observed functional annotations, and a residual that up-weights cold-start positives did not improve "
        "cold-start recovery without a loss in overall performance. SARA therefore falls back directly to retrieval "
        "evidence when semantics are unavailable and passes the missingness to the ranker explicitly as NaN rather than "
        "filling it with predicted values.",
    ])


def note_s5(R, N) -> str:
    hc = R["historical_contrasts"]
    hp = N["x"]["hp"]
    x2, x1 = hp["x2"], hp["x1"]
    return "\n\n".join([
        "## Supplementary Note S5. Design and results of temporal extrapolation",
        "Temporal extrapolation uses 2023-12-31 as the cut-off. The inputs available before the cut-off, with their dates "
        "and hashes, are listed in Supplementary Table S9; inputs that are dated after the cut-off and cannot be "
        "backdated (the gene background and gene index, the representative protein sequences, the disease equivalence "
        "groups and family assignments, and the hyper-parameters and frozen encoders of SARA) are disclosed item by item "
        "below Supplementary Table S9. The pre-cut-off inputs were used to rebuild the disease text, the HPO space, the "
        "source-excluded gene–phenotype annotations, the TF-IDF vectors, the ontology and the GO functional anchors, and "
        "all models were retrained with a single setting independent of the outer contexts (the most frequent choice "
        "among the five outer contexts). The new-association task evaluates the associations added after the cut-off "
        "for diseases that already had associations at the cut-off (genes known before the cut-off being removed from "
        "the ranking), and the new-disease task evaluates diseases without any association at the cut-off.",
        f"On the new-association task ({i0(hc['x2_deployed_vs_channel_lgb_v6']['n'])} diseases, {i0(x2['pairs'])} pairs), "
        "the Recall@10 of SARA, retrieval-channel LightGBM, the base ranker and stand-alone semantic retrieval were "
        f"{num(hc['x2_deployed_vs_channel_lgb_v6']['a'])}, {num(hc['x2_deployed_vs_channel_lgb_v6']['b'])}, "
        f"{num(hc['x2_deployed_vs_B']['b'])} and {num(hc['x2_deployed_vs_standalone_sF']['b'])}, respectively; on the "
        f"new-disease task ({i0(hc['x1_deployed_vs_channel_lgb_v6']['n'])} diseases, {i0(x1['pairs'])} pairs), they were "
        f"{num(hc['x1_deployed_vs_channel_lgb_v6']['a'])}, {num(hc['x1_deployed_vs_channel_lgb_v6']['b'])}, "
        f"{num(hc['x1_deployed_vs_B']['b'])} and {num(hc['x1_deployed_vs_standalone_sF']['b'])}. SARA and "
        f"retrieval-channel LightGBM performed comparably on both tasks ({cin(hc['x2_deployed_vs_channel_lgb_v6'])}; "
        f"{cin(hc['x1_deployed_vs_channel_lgb_v6'])}), and SARA outperformed stand-alone semantic retrieval on the "
        f"new-disease task ({cin(hc['x1_deployed_vs_standalone_sF'])}). In the pairwise comparison of the "
        f"{i0(x2['pairs'])} pairs of the new-association task, SARA ranked {i0(x2['deployed_better'])} pairs higher and "
        f"retrieval-channel LightGBM {i0(x2['channel_better'])}, and {i0(x2['tie'])} pairs were tied "
        f"({i0(x2['both_unlisted'])} of them listed by neither method); SARA brought {i0(x2['listed_deployed'])} pairs "
        f"into the candidates and retrieval-channel LightGBM {i0(x2['listed_channel'])} (Figure 9c, f of the main text).",
    ])


def note_s6(R, N) -> str:
    h = R["main"]["hierarchy"]
    return "\n\n".join([
        "## Supplementary Note S6. Statistical analysis and computing resources",
        "The primary comparison — the end-to-end Recall@10 difference between SARA and retrieval-channel LightGBM — was "
        "fixed before any test evaluation, and the lower bound of its 95% CI was above zero "
        f"({cin(h['h1_vs_channel_lgb_v6'])}). For the comparison with multimodal direct fusion, a non-inferiority margin "
        f"of −0.005 was specified in advance: the observed difference was {ci(h['h2_vs_v71_direct_strict'], d=4)}, and "
        "because the lower bound of the interval lies slightly below the margin, non-inferiority was not established at "
        "this margin; as the point estimate is close to zero and the interval includes zero, the main text describes the "
        "two methods as performing comparably. The difference in cold-start recovery from multimodal direct fusion "
        f"({cin(h['h3_cold_vs_v71_direct_strict'], d=4)}) is reported descriptively only. All other comparisons and all "
        "stratified and explanatory analyses report only estimates and intervals, without significance testing, so no "
        "correction for multiple comparisons was applied. The 5 × 864 development evaluations are correlated repetitions "
        "used only for selection and are not treated as independent samples.",
        "All intervals are bootstrap percentile intervals from resampling disease families: 10,000 resamples for the "
        "primary comparison (seed 20260923) and 1,000–2,000 for descriptive intervals; association-level proportions "
        "resample numerators and denominators jointly, with families as clusters. Per-disease metrics are first averaged "
        "over the three seeds and then macro-averaged over diseases; the union of hits over seeds is never used.",
        f"**Computing resources.** All {i0(N['runs'])} fitting and evaluation runs are recorded with content hashes; the "
        f"timed runs took about {num(N['timed_h'], 1)} h in total, and the peak GPU memory and peak RAM of a single run "
        f"were {num(N['gpu_peak'] / 1024, 2)} GiB and {num(N['rss_peak'] / 1024, 1)} GiB (Supplementary Figure S6).",
    ])


# ============================================================================ tables
def table_s1(R, N) -> str:
    dis = pd.read_parquet(PUB / "diseases.parquet")
    t = dis[dis["role"] == "test_fold"]
    ctl = S.J(paths8.MANIFESTS / "dev_teacher_controls_v8.json")["per_context"]
    rows_a, rows_b = [], []
    for j in range(5):
        s = S._sel(j)
        tt = t[t["fold"] == j]
        tr = S._teacher(j)
        bc = max(s["summary"]["corrector"], key=lambda r: r["dev_recall10"])["dev_recall10"]
        rows_a.append([f"Outer {j}", f"$F_{j}$ / $F_{(j + 1) % 5}$", i0(len(tt)), i0(tt["n_rel"].sum()),
                       i0(tt["family_group"].nunique()), i0(tr["fit"]["n_reference"])])
        rows_b.append([f"Outer {j}", "Full background" if ctl[f"f{j}_outer"]["controls"] == "background"
                       else "Reference-associated genes", num(s["w"]["w"], 1), str(tr["fit"]["best_epoch"]),
                       num(s["summary"]["B_dev"]["dev_recall10"]), num(bc), num(s["ab"]["best_A_dev_recall10"]),
                       _cfg(s["ab"]["best_A"])])
    return ("Table S1. Size of the outer contexts and development-set choices (seed 42; choices use only the development "
            "set and the $F_{j+1}$ of the same context)\n\n(a) Size\n\n" +
            table_md(["Context", "Test / validation fold", "Test diseases", "Test associations", "Test families",
                      "Reference diseases"], rows_a) +
            "\n\n(b) Development-set choices (R@10 on the development set; configurations are given as number of leaves / "
            "minimum samples per leaf)\n\n" +
            table_md(["Context", "Unlabeled controls", "$w_j$", "Best epoch of the semantic model", "Base ranker",
                      "Best corrector", "SARA ranker", "Selected configuration"], rows_b))


def table_s2(R, N) -> str:
    from .pubdata8 import feat_family
    fi = pd.read_parquet(PUB / "feature_importance.parquet")
    agg = fi.groupby("feature")["gain_share"].agg(["mean", "std"])
    order = list(fi["feature"].drop_duplicates())
    rows = []
    for fam in ["V6 检索通道", "疾病描述", "候选融合", "语义", "表达", "基因可用性"]:
        for f in [x for x in order if feat_family(x) == fam]:
            rows.append([FAMILY[fam], f"`{f}`", cap(FEATURE[f]),
                         f"{pc(agg.loc[f, 'mean'], 2)} ± {pc(agg.loc[f, 'std'], 2)}"])
    return ("Table S2. The 49 features of the SARA ranker (gain share: mean ± s.d. over 15 models)\n\n" +
            table_md(["Feature family", "Feature", "Description", "Gain share"], rows))


def table_s3(R, N) -> str:
    comp = R["comparators"]
    rows_a = [
        ["Retrieval-channel LightGBM", "30 retrieval features", "Own retrieval top 500",
         "LambdaRank, early stopping on $F_{j+1}$", "Primary comparator"],
        ["Base ranker", "30 retrieval features (out-of-fold)", "SARA candidates", "As above",
         "Retrieval baseline on the same candidates"],
        ["Fixed-weight fusion", "Fixed weights on retrieval-source scores", "Retrieval top 500; SARA candidates",
         "No training (weights selected on the development set)", "Rule-based baseline"],
        ["Multimodal direct fusion", "30 retrieval features + 14 multimodal semantic-core features (missing set to 0)",
         "Retrieval top 500; SARA candidates", "LambdaRank, early stopping on $F_{j+1}$", "Multimodal comparator"],
        ["Stand-alone semantic rankings", "$s_F$, core score $r$, sequence score", "Full background (top 500)",
         "Semantic model, checkpoint chosen by $s_F$", "Semantic ablation"],
        ["Untrained SapBERT similarity", "Unprojected disease–function SapBERT similarity", "Full background",
         "No training", "Semantic ablation"],
        ["Linear two-tower model", "Same inputs as the semantic model, linear projections", "Full background",
         "Contrastive learning, epoch chosen on $F_{j+1}$", "Semantic ablation"],
        ["KDGene [@wang2024kdgene]", "Knowledge-graph embedding (1,000-dimensional entities and relations)",
         "SARA candidates", "At most 100 epochs, patience 10, epoch chosen on $F_{j+1}$ with validation diseases unseen",
         f"External method; R@10 {num(comp['kdgene']['recall10'])}"],
        ["SIGN-style graph model [@frasca2020]", "Context-view propagation of orders 0–2 + two-layer decoder",
         "SARA candidates", "AdamW (learning rate 10⁻³, weight decay 10⁻⁴), 30 epochs, epoch chosen on $F_{j+1}$",
         f"External method; R@10 {num(comp['sign']['recall10'])}"],
        ["ProtST zero-shot [@xu2023protst]", "Zero-shot text–protein retrieval (pretrained weights only)",
         "Full background (protein-coding)", "No training", "Open-knowledge control"],
    ]
    a = table_md(["Method", "Input", "Candidate list", "Training and checkpoint", "Role"], rows_a)
    rows_b = []
    for r in R["semantic"]["table"]:
        rows_b.append([SEMANTIC.get(r["method"], r["method"]), num(r["recall10"]), num(r["recall50"]),
                       num(r["cold_recall10"]), num(r["coverage500"]),
                       ci(r["vs_sF"]) if isinstance(r.get("vs_sF"), dict) else "—"])
    b = table_md(["Method", "R@10", "R@50", "$R^c$@10", "Coverage@500", "Difference from $s_F$ (segmented text)"], rows_b)
    k, s = comp["kdgene"]["deployed_minus"], comp["sign"]["deployed_minus"]
    return ("Table S3. Settings of the comparison methods and performance of full-background semantic rankings\n\n"
            "(a) Comparison methods\n\n" + a + "\n\n" +
            "Note: KDGene and the SIGN-style graph model were retrained on the same candidate lists under the out-of-fold "
            f"rules (seed 42); the Recall@10 differences of SARA from them were {ci(k)} and {ci(s)}, respectively."
            "\n\n(b) Full-background semantic rankings (seed 42; each method ranks all background genes and is evaluated "
            "on its top 500)\n\n" + b)


def table_s4(R, N) -> str:
    st = R["strata"]["strata"]
    zs = R["strata"]["zero_signal_s42"]
    ms = ["deployed", "channel_lgb_v6", "v71_direct_strict"]
    items = [("single_relation_recall10", "Single-association diseases, R@10", num),
             ("multi_relation_recall10", "Multi-association diseases, R@10", num),
             ("omim_mapped_recall10", "With OMIM mapping, R@10", num),
             ("not_omim_recall10", "Without OMIM mapping, R@10", num),
             ("burden_1", "Diseases with 1 positive, R@10", num),
             ("burden_2_5", "Diseases with 2–5 positives, R@10", num),
             ("burden_gt5", "Diseases with >5 positives, R@10", num),
             ("therapeutic_only_hit10_micro",
              f"Top-10 hit rate of therapeutic associations ({i0(st['deployed']['therapeutic_only_pairs'])} pairs)", pc)]
    rows = [[lab] + [fn(st[m][k]) for m in ms] for k, lab, fn in items]
    a = table_md(["Stratum", "SARA", "Retrieval-channel LightGBM", "Multimodal direct fusion"], rows)
    sg = {r["method"]: r for r in R["subgroups"]["table"]}
    ms2 = [m for m in ("deployed", "channel_lgb_v6", "B", "v71_direct_strict") if m in sg]
    it2 = [("ncrna", "Non-coding RNA"), ("nogo_protein", "Protein-coding without GO"),
           ("sem_unavail", "Semantically unavailable"), ("mask_test", "Test-masked genes")]
    rows2 = []
    for k, lab in it2:
        q = sg["deployed"][f"{k}_queries"]
        rows2.append([f"{lab} ({i0(q)})"] + [f"{pc(sg[m][f'{k}_pool_micro'])} / {pc(sg[m][f'{k}_hit10_micro'])}"
                                             for m in ms2])
    b = table_md(["Subgroup (query associations)"] + [method(m, table=True) for m in ms2], rows2)
    return ("Table S4. Retrieval performance by disease stratum and information-missing subgroup (mean of three seeds)"
            "\n\n(a) Strata by disease characteristics\n\n" + a +
            "\n\n(b) Information-missing subgroups: share entering the candidates / share entering the top 10 "
            "(association-level micro-average)\n\n" + b +
            f"\n\nNote: Of the {i0(zs['queries'])} test associations of seed 42, {i0(zs['no_v6_signal'])} have no signal "
            f"from any retrieval source, {i0(zs['no_semantic'])} are semantically unavailable and "
            f"{i0(zs['zero_signal_both'])} ({pc(zs['zero_signal_share'])}) have neither; none of the latter enters the "
            "candidates.")


def table_s5(R, N) -> str:
    d3 = S.sd("Fig3_main_results", "d")
    rows = []
    for (f, s), g in d3.groupby(["fold", "seed"]):
        a = g[g["vs"] == "channel_lgb_v6"].iloc[0]
        b = g[g["vs"] == "v71_direct_strict"].iloc[0]
        rows.append([str(f), str(s), i0(a["n"]),
                     ci({"delta": a["delta"], "ci_low": a["lo"], "ci_high": a["hi"]}, label=False),
                     ci({"delta": b["delta"], "ci_low": b["lo"], "ci_high": b["hi"]}, label=False)])
    a_ = table_md(["Outer fold", "Seed", "Test diseases", "SARA − retrieval-channel LightGBM",
                   "SARA − multimodal direct fusion"], rows)
    v = R["variants"]
    vr = [("v6pool", "SARA ranker on the retrieval top 500"), ("plain_rrf", "Standard RRF candidates (500)"),
          ("bridge950", "Extended candidates (~950)"), ("expr_off", "Expression features off"),
          ("expr_random", "Random tissue expression")]
    rows_b = []
    for k, lab in vr:
        d = v[k]["deployed"]
        rows_b.append([lab, num(d["recall10"]), num(d["cold_recall10"]), pc(d["coverage500"]), pc(d["ncrna_pool_micro"]),
                       ci(d["delta_vs_main"], label=False)])
    b_ = table_md(["Variant", "R@10", "$R^c$@10", "Coverage", "ncRNA positives listed",
                   "R@10 difference from SARA (95% CI)"], rows_b)
    ref = v["plain_rrf"]["deployed"]["delta_vs_main"]["b"]
    return ("Table S5. Robustness across outer-fold × seed combinations and candidate-list variants\n\n"
            "(a) Recall@10 differences (95% CI) in the 15 outer-fold × seed combinations\n\n" + a_ +
            "\n\n(b) Variants of the candidate list and the expression features (seed 42; every variant retrained end to "
            f"end; R@10 of SARA, {num(ref)})\n\n" + b_)


def table_s6(R, N) -> str:
    d = N["s5d"]
    rows_a = [[mesh_table(r["name"], r["code"]), i0(r["n"]), num(r["deployed"]), num(r["v71_direct_strict"]),
               num(r["channel_lgb_v6"]), num(r["standalone_sF"])] for _, r in d.iterrows()]
    rows_b = [[mesh_table(r["name"], r["code"]),
               ci({"delta": r["delta"], "ci_low": r["lo"], "ci_high": r["hi"]}, label=False),
               ci({"delta": r["delta_v71"], "ci_low": r["lo_v71"], "ci_high": r["hi_v71"]}, label=False)]
              for _, r in d.iterrows()]
    return ("Table S6. Recall@10 by MeSH disease category (categories with at least 60 test diseases; mean of three "
            "seeds)\n\n(a) Recall@10 of four methods\n\n" +
            table_md(["Category", "Diseases", "SARA", "Multimodal direct fusion", "Retrieval-channel LightGBM",
                      "Semantic retrieval $s_F$"], rows_a) +
            "\n\n(b) Differences in Recall@10 (95% CI)\n\n" +
            table_md(["Category", "SARA − retrieval-channel LightGBM", "SARA − multimodal direct fusion"], rows_b))


def table_s7(R, N) -> str:
    e4 = R["e4"]
    c8 = S.sd("Fig8_evidence_sources", "c").set_index("index")
    rows = []
    for (k, _, den), (k2, lab, _) in zip(S.E4_PUB, E4):
        assert k == k2
        a = e4[k]
        red, dif = a["new_minus_old_reduced"], a["difference"]
        if den is not None and den in c8.index:
            dd = c8.loc[den]
            dtxt = " / ".join(pc(float(dd[c]), 2 if float(dd[c]) < 0.01 else 1)
                              for c in ("annotation_rows", "genes", "query_relations"))
        else:
            dtxt = "As the row above (size-matched)"
        rows.append([lab, dtxt, num(red["a"]), num(red["b"]), sgn(red["delta"]), ci(dif, label=False)])
    full = e4["e4_layer1"]["new_minus_old_full"]
    return ("Table S7. Retrieval performance under each evidence-deletion condition (seed 42, full-pipeline retraining; "
            "re-scoring conditions replace only the inference inputs)\n\n" +
            table_md(["Condition", "Deleted share", "SARA R@10", "Retrieval-channel LightGBM R@10",
                      "SARA advantage (reduced data)", "Change in advantage (reduced − full, 95% CI)"], rows) +
            "\n\nNote: Deleted share is given as annotation rows / affected genes / affected query associations. With the "
            f"full data, the advantage of SARA is {ci(full)}. * Uses the literature of evaluation associations; "
            "sensitivity analysis only, not used in any model selection.")


def table_s8(R, N) -> str:
    e2 = {r["method"]: r for r in R["e2"]["table"]}
    con = R["e2"]["contrasts"]
    rows = []
    for b in (0, 1, 2):
        m = f"corrector_beta{b}"
        rows.append([f"Gated corrector ($\\beta={b}$)", num(e2[m]["recall10"]), num(e2[m]["cold_recall10"]),
                     num(e2[m]["cold_to_all10"]), ci(con[m]["recall10_vs_deployed"], label=False),
                     ci(con[m]["cold_vs_deployed"], label=False)])
    d = e2["deployed"]
    rows.insert(0, ["SARA", num(d["recall10"]), num(d["cold_recall10"]), num(d["cold_to_all10"]), "Reference",
                    "Reference"])
    a = table_md(["Model", "R@10", "$R^c$@10", "$R^{c\\to all}$@10", "ΔR@10 (95% CI)", "Δ$R^c$@10 (95% CI)"], rows)
    v = R["variants"]
    rows_b = []
    for (k, _), (k2, lab) in zip(S.E5_PUB, E5):
        assert k == k2
        dd = v[k]["deployed"]
        rows_b.append([lab, num(dd["recall10"]), num(dd["cold_recall10"]), pc(dd["mask_test_pool_micro"]),
                       pc(dd["mask_test_hit10_micro"]), ci(dd["delta_vs_main"], label=False)])
    b = table_md(["Condition (test-masked genes)", "R@10", "$R^c$@10", "Masked-gene positives listed",
                  "Masked-gene positives in top 10", "R@10 difference from the full model (95% CI)"], rows_b)
    return ("Table S8. Retrieval performance of the two alternative designs\n\n"
            "(a) Gated cold-balance corrector (mean of three seeds; retrained on the SARA candidates, $\\lambda$ selected "
            "on the development set; $R^{c\\to all}$@10 is the number of cold-start positives in the top 10 as a share of "
            "all test positives)\n\n" + a +
            "\n\n(b) Sequence imputation model (seed 42; full-pipeline retraining after removal of the GO annotations of "
            "test-masked genes; the last row switches imputation on with all GO annotations retained)\n\n" + b)


def table_s9(R, N) -> str:
    et = pd.DataFrame(R["external"]["table"])
    subs = [("union", "Union"), ("GenCC", "GenCC"), ("ClinVar", "ClinVar"),
            ("independent_evidence", "Literature-independent subset"),
            ("no_neighbour_relation", "No same association in neighboring diseases")]
    ms = [m for m in ("deployed", "v71_direct_strict", "channel_lgb_v6", "B", "F7_v6") if m in set(et["method"])]
    rows = []
    for s, lab in subs:
        sizes = set()
        for m in ms:
            r = _row(et, method=m, subset=s)
            if not r:
                continue
            first = not sizes
            sizes.add((r["n_diseases"], r["n_pairs"]))
            rows.append([f"{lab} ({i0(r['n_diseases'])} diseases / {i0(r['n_pairs'])} pairs)" if first else "",
                         method(m, table=True), num(r["recall10"]), num(r["hit10"]), num(r["recall50"]),
                         pc(r["in_list"])])
        assert len(sizes) == 1, f"subset {s}: sizes differ between methods"
    a = table_md(["Subset", "Method", "R@10", "Hit@10", "R@50", "Listed"], rows)
    hi = S.J(paths8.AUDIT / "historical_inputs_v8.json")
    rows_b = [[HIST[r["input"]], r["date"], "Yes" if r["not_later_than_cutoff"] else "No", (r["sha256"] or "")[:12]]
              for r in hi["inputs"]]
    b = table_md(["Input", "Date", "Not later than the cut-off", "SHA-256 (first 12 characters)"], rows_b)
    G = i0(N["n_genes"])
    later = "; ".join(f"{LATER[x['input']].replace('19,861', G)} ({x['date']}, {ROLE[x['role']]})"
                      for x in hi.get("later_inputs_disclosed", []))
    return ("Table S9. Evaluation on external labels and input versions for temporal extrapolation\n\n"
            "(a) Performance on external labels (filtered ranking, mean of three seeds; ‘Listed’ is the share of external "
            "associations entering the candidate list; ‘No same association in neighboring diseases’ is the subset whose "
            "association appears in none of the parent, child or same-family reference diseases)\n\n" + a +
            f"\n\n(b) Inputs of temporal extrapolation (cut-off {hi['cutoff']})\n\n" + b +
            f"\n\nNote: Inputs dated after the cut-off, disclosed item by item: {later}.")


def table_s10(R, N) -> str:
    c5 = pd.DataFrame(R["support"]["cohort533"])
    rows = [[SUPPORT[r.method], str(int(r.k)), num(r.recall10), num(r.recall50), i0(r.n_diseases)]
            for r in c5.itertuples()]
    a = table_md(["Method", "$k$", "R@10", "R@50", "Diseases"], rows)
    c1 = pd.DataFrame(R["support"]["cohort1shot"])
    rows1 = [[SUPPORT[r.method], num(r.recall10), num(r.recall50), i0(r.n_diseases)] for r in c1.itertuples()]
    b = table_md(["Method", "R@10", "R@50", "Diseases"], rows1)
    t = pd.read_csv(paths8.EXT_RES / "expression_v8_table.csv")
    t["base"] = t["set"].str.rsplit("_top", n=1).str[0]
    t["K"] = t["set"].str.rsplit("_top", n=1).str[1]
    cols = [(acc, c) for acc, c in t[["accession", "contrast"]].drop_duplicates().itertuples(index=False)]
    rows_c = []
    for base, lab in EXPR_SET:
        for K in ("20", "50"):
            s = t[(t["base"] == base) & (t["K"] == K)]
            if s.empty:
                continue
            cells = []
            for acc, c in cols:
                r = s[(s["accession"] == acc) & (s["contrast"] == c)]
                if r.empty:
                    cells.append("—")
                    continue
                r = r.iloc[0]
                p = float(r["p_value"])
                bold = "**" if p < 0.05 else ""
                cells.append(f"{num(r['enrichment'])} ({bold}{num(p, 3) if p >= 0.001 else sci(p)}{bold})")
            rows_c.append([lab if K == "20" else "", K] + cells)
    c_ = table_md(["Gene set", "Top"] + [f"{acc} {CONTRAST[c]}" for acc, c in cols], rows_c)
    cr = N["case_rows"]
    rows_d = []
    for c in N["cases"]:
        for k, m in enumerate(("deployed", "channel_lgb_v6")):
            s_ = cr[(cr["disease_id"] == c["disease_id"]) & (cr["method"] == m)].sort_values("rank")
            rows_d.append([f"{cap(CASE[c['disease_id']])} ({c['disease_id']})" if k == 0 else "", method(m, table=True),
                           ", ".join((f"**{x}**" if h else x) + ("°" if cold else "")
                                     for x, h, cold in zip(s_["symbol"], s_["held_out_positive"],
                                                           s_["reference_path_cold"]))])
    d_ = table_md(["Disease", "Method", "Top 10"], rows_d)
    return ("Table S10. Evaluation of the support tasks, the expression cohorts and the case diseases\n\n"
            "(a) Support task (533 diseases; $k$ is the number of revealed support genes; combination weight of semantics "
            "and RWR $\\omega=0.25$)\n\n" + a +
            "\n\n(b) One-shot support task (760 diseases, $k=1$)\n\n" + b +
            "\n\n(c) Rank tests of the frozen gene sets in four expression cohorts (enrichment, with the *P* value in "
            "parentheses; enrichment is the probability that the $\\left|t\\right|$ of a selected gene exceeds that of matched "
            "background genes; bold marks $P<0.05$)\n\n" + c_ +
            "\n\n(d) Top 10 genes of the case diseases (seed 42; bold, held-out true associations; °, cold-start "
            "positives)\n\n" + d_)


# ============================================================================ figure legends
CI_NOTE = "Intervals are bootstrap 95% CIs from resampling disease families."
SFIGS = [
    ("S1", "FigS1_teacher_training", "Training of the semantic model and selection of the unlabeled control set",
     "**a** Full-background $s_F$ Recall@10 of the semantic model on the $F_{j+1}$ of the same context as a function of "
     "the training epoch in the five outer contexts (shade, seed; gray lines, reference-associated-gene controls, seed "
     "42); dots mark the best epochs. **b** Left, $s_F$ Recall@10 of the two unlabeled control sets on the development "
     "set; right, best (filled) and early-stopping (open) epochs of the semantic model in each outer context."),
    ("S2", "FigS2_development_selection", "Configuration selection on the development set",
     "**a** Development positive coverage@500 as a function of the RRF weight $w$ in each outer context (solid, "
     "availability-aware RRF; dashed, standard RRF; orange, retrieval top 500; black circles, the selected $w_j$). **b** "
     "Development Recall@10 of the six candidate configurations of the SARA ranker (number of leaves × minimum samples "
     "per leaf); black frames mark the configuration selected for each outer context."),
    ("S3", "FigS3_fold_seed_stability", "Stability of performance across outer-fold × seed combinations",
     "**a** End-to-end Recall@10 of each outer fold × seed: rows are methods, columns are outer fold · seed and the "
     "rightmost column gives the row means (common color scale). **b** Per-disease comparisons by outer fold (three-seed "
     "mean): left, shares of wins, ties and losses of SARA against retrieval-channel LightGBM and multimodal direct "
     "fusion; right, differences with 95% CIs (filled circles, against retrieval-channel LightGBM; open squares, against "
     "multimodal direct fusion). **c** Agreement of the per-disease Recall@10 of SARA between seeds (hexagonal bins, log "
     "color scale; $r$ is the Pearson correlation coefficient). " + CI_NOTE),
    ("S4", "FigS4_semantic_variants", "Comparison of full-background semantic rankings and disease-text variants",
     "**a** Recall@10 (bars) and $R^c$@10 (circles) of each semantic method; purple marks the three text versions of "
     "semantic retrieval $s_F$. **b** Recall@10 differences from semantic retrieval $s_F$ (segmented text) with 95% CIs "
     "(rows as in panel a)."),
    ("S5", "FigS5_embedding", "Structure of the disease–function semantic space",
     "**a** t-SNE of the disease vectors $z_d$ of the semantic model (outer fold 0, seed 42); each small panel highlights "
     "one MeSH category (numbers of diseases in parentheses). **b** Cosine distributions of known disease–gene "
     "associations and of random genes with GO annotations; these associations were visible during training and serve "
     "only to illustrate the structure of the space."),
    ("S6", "FigS6_compute_resources", "Use of computing resources",
     "**a** Completion timeline of the runs of each component (point size proportional to the timed seconds). **b** "
     "Number of runs and timed machine hours of each component."),
    ("S7", "FigS7_evidence_blocks",
     "Recovery of evidence-block associations and the performance advantage of SARA under evidence-block isolation",
     "**a** Top-10 recovery of the test associations in the isolated evidence blocks: open symbols, full data; arrows "
     "point to the reduced data (blue, SARA; orange, retrieval-channel LightGBM, retrained only in the evidence-block "
     "condition). **b** Recall@10 difference between SARA and retrieval-channel LightGBM under each deletion condition; "
     "the vertical line and the gray band give the estimate and its 95% CI with the full data."),
    ("S8", "FigS8_alternative_designs", "Development and test evaluation of the two alternative designs",
     "**a** Validation Recall@10 of the gated cold-balance corrector on the $F_{j+1}$ of each outer context as a "
     "function of boosting rounds (development stage, seed 42; shade indicates $\\beta$, solid and dotted lines indicate "
     "$\\lambda$); dots mark the best rounds. **b** Model selection on the development set: blue, the six configurations "
     "of the SARA ranker (black circles, the selected configurations); magenta, correctors (shape indicates $\\beta$, "
     "filled and open symbols indicate $\\lambda$); yellow, base ranker; the light-blue band marks the range within 0.005 "
     "of the best ranker. **c** Differences between the correctors with three values of $\\beta$ and SARA on the test "
     "set (three seeds; $\\lambda$ selected on the development set): left, Recall@10; right, $R^c$@10 and the comparison "
     "of cold-start recovery between values of $\\beta$. **d** Effect of the sequence imputation model on the "
     "development mask set: change in the share of masked-gene positives entering the candidates relative to no "
     "imputation (filled, five outer contexts) and the control with pairings shuffled within length strata (open); the "
     "light-blue area is the positive region required for adoption. **e** Quality of the imputed outputs for masked "
     "genes: left, top-10 hit rate of retrieving the gene's own vector among about 19,000 true functional vectors; "
     "right, cosine with the true vector (filled) and mean pairwise cosine (open). **f** After removal of the GO "
     "annotations of test-masked genes, the shares of masked-gene positives entering the candidates and the top 10 under "
     "each imputation variant, and the difference in overall Recall@10 from the full model (seed 42); vertical lines "
     "mark the level without imputation. **g** Pairwise rank changes of each imputation variant relative to no "
     "imputation: rightwards, rank up; leftwards, rank down; dark narrow bars, positives crossing the listing boundary. "
     + CI_NOTE),
]


# ============================================================================ assembly
def compose(R=None, N=None) -> tuple[str, dict]:
    from .refs_en import resolve
    if R is None:
        R, N = numbers9()
    TRACE.clear()
    head = ["# Supplementary material", f"**{TITLE}**",
            "This supplementary material contains Supplementary Notes S1–S6, Supplementary Tables S1–S10 and "
            "Supplementary Figures S1–S8. Figures, tables, equations and sections numbered without ‘S’ refer to the main "
            "text."]
    notes = [note_s1(R, N), note_s2(R, N), note_s3(R, N), note_s4(R, N), note_s5(R, N), note_s6(R, N)]
    tabs = ["## Supplementary tables", table_s1(R, N), table_s2(R, N), table_s3(R, N), table_s4(R, N), table_s5(R, N),
            table_s6(R, N), table_s7(R, N), table_s8(R, N), table_s9(R, N), table_s10(R, N)]
    figs = ["## Supplementary figures"]
    for n, stem, title, leg in SFIGS:
        figs += [f"<!-- figure {stem} -->", f"**Figure {n}. {title}.** {leg}"]
    md = "\n\n".join(p.strip("\n") for p in head + notes + tabs + figs + ["## Supplementary references",
                                                                          "<<REFERENCES>>"]) + "\n"
    md, eqs = number_equations(md)
    md, refs, order, recs = resolve(md)
    md = typography(md.replace("<<REFERENCES>>", "\n".join(refs)))
    return md, {"stem": SUPP_STEM, "ref_order": order, "n_refs": len(refs), "equations": eqs,
                "figures": [(n, s) for n, s, _, _ in SFIGS], "trace": {k: v for k, v in TRACE.items()}}


def write() -> dict:
    R, N = numbers9()
    md, info = compose(R, N)
    EN.mkdir(parents=True, exist_ok=True)
    p = EN / f"{SUPP_STEM}.md"
    p.write_text(md, encoding="utf-8")
    (ROOT / "build" / "en").mkdir(parents=True, exist_ok=True)
    (ROOT / "build" / "en" / "supp_en_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=1),
                                                             encoding="utf-8")
    return {"md": str(p), "chars": len(md), "refs": info["n_refs"], "equations": len(info["equations"])}


if __name__ == "__main__":
    print(json.dumps(write(), ensure_ascii=False, indent=1))

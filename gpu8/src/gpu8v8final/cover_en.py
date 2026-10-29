"""Draft cover letter of the English SARA submission (Briefings in Bioinformatics).

Every number is formatted by fmt_en from the same results as the manuscript (traced in fmt9.TRACE); author, journal
article type, conflicts and AI-use statements are placeholders for the authors to complete.
"""
from __future__ import annotations

import json

import pandas as pd

from .build_docx_en import BUILD, EnBuilder
from .fmt_en import TRACE, cin, i0, num, pc
from .manuscript9 import _row
from .manuscript_en import DATE, EN, TITLE, typography
from .numbers9 import numbers9

STEM = f"SARA_cover_letter_{DATE}"


def compose(R=None, N=None) -> tuple[str, dict]:
    if R is None:
        R, N = numbers9()
    TRACE.clear()
    z = N["z"]
    dep, ch = z["table"]["deployed"], z["table"]["channel_lgb_v6"]
    h = z["h"]
    eu = _row(pd.DataFrame(z["ext"]["contrasts"]), b="channel_lgb_v6", subset="union")
    paras = [
        "[Date]",
        "The Editors  \n*Briefings in Bioinformatics*",
        "Dear Editors,",
        f"We are pleased to submit our manuscript entitled “{TITLE}” for consideration for publication in *Briefings "
        "in Bioinformatics* as a [article type].",
        "Knowledge of disease–gene associations is highly concentrated: in our data, the 10% of genes with the most "
        f"associations hold {pc(N['top10'], 0)} of all known associations, and only {pc(N['all6'] / N['n_genes'])} of "
        "genes have all six information layers. Retrieval methods that treat missing "
        "information as a poor match therefore systematically under-rank poorly studied genes, which are precisely the "
        "candidates for which computational prioritization is most needed.",
        "SARA addresses this imbalance with three components: (i) a disease–function semantic model that provides "
        "evidence independent of known associations; (ii) an availability-aware reciprocal rank fusion that reduces "
        "exactly to the retrieval order when semantic information is unavailable, so that candidates lacking a modality "
        "are not penalized; and (iii) a learning-to-rank model with explicitly encoded missing values.",
        f"Under family-disjoint five-fold cross-validation on {i0(dep['n_diseases'])} test diseases, SARA raised the "
        f"disease-macro-averaged Recall@10 from {num(ch['recall10'])} to {num(dep['recall10'])} "
        f"({cin(h['h1_vs_channel_lgb_v6'])}); the difference was positive in all 15 outer-fold × seed combinations, and "
        f"the top-10 recovery of cold-start positives rose from {num(ch['cold_recall10'])} to "
        f"{num(dep['cold_recall10'])}. The candidate budget needed to cover half of the test positives fell from about "
        f"{i0(N['K50_rB'])} to about {i0(N['K50_r_aware'])}. SARA performed comparably to a multimodal direct-fusion "
        "model and showed a consistent direction of benefit on external GenCC and ClinVar labels (difference "
        f"{cin(eu)}), in a one-shot support task and in four independent transcriptomic cohorts.",
        "We believe that the evaluation design will be of particular interest to the readership of *Briefings in "
        "Bioinformatics*: all data splits are family-disjoint, all second-level features are generated out-of-fold, a "
        "three-layer evidence-deletion audit with matched random controls shows that the gains do not depend on "
        "functional annotations that share literature with the evaluated associations, and TreeSHAP explanations show "
        "where semantic evidence takes over from retrieval evidence. The approach is therefore relevant both to disease "
        "gene prioritization and, more broadly, to learning from heterogeneous and incomplete biomedical knowledge.",
        "This manuscript is original, has not been published previously and is not under consideration for publication "
        "elsewhere. All authors have read and approved the submission. [The authors declare no competing interests.] The "
        "code, the frozen model configurations and the plotting data of every figure panel will be made publicly "
        "available upon acceptance. [Statement on the use of artificial intelligence tools, if applicable.]",
        "[Optional: suggested reviewers with their affiliations and e-mail addresses; reviewers to be excluded.]",
        "Thank you for considering our work. We look forward to hearing from you.",
        "Sincerely,",
        "[Corresponding author name]  \n[Position, department, institution]  \n[Postal address]  \n[E-mail address]",
    ]
    md = typography("\n\n".join(paras) + "\n")
    return md, {"stem": STEM, "trace": {k: v for k, v in TRACE.items()}}


def write() -> dict:
    R, N = numbers9()
    md, info = compose(R, N)
    p = BUILD / f"{STEM}.md"
    BUILD.mkdir(parents=True, exist_ok=True)
    p.write_text(md, encoding="utf-8")
    (BUILD / "cover_en_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    b = EnBuilder("notes", {}, line_numbers=False)
    st = b.build(p, EN / f"{STEM}.docx", images=False)
    ok = (st["a4"] and st["page_field"] and st["omml_objects"] == 0 and st["ole_objects"] == 0
          and not st["leftover_dollar"] and not st["leftover_markup"])
    return {"docx": st["path"], "ok": ok, "paragraphs": st["paragraphs"], "words": len(md.split())}


if __name__ == "__main__":
    print(json.dumps(write(), ensure_ascii=False, indent=1))

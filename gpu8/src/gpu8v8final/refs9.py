"""Reference registry, Crossref verification and citation resolution for the SARA manuscript.

Text cites with ``[@key]`` or ``[@key1; @key2]``.  ``resolve(md)`` numbers the references in order
of first citation, rewrites the markers to ``[1]``, ``[2, 3]`` or ``[4–6]`` and returns the numbered
Vancouver-style list.  ``verify()`` checks every DOI against Crossref (title, first author, year,
journal); entries without a DOI are checked by a bibliographic query or marked as conference/book
items.  Only verified entries may be cited (enforced in verify9).
"""
from __future__ import annotations

import difflib
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

CACHE = Path(r"D:\信\gpu8\submission9\refs_cache.json")

# key: (vancouver text without DOI, doi or None, kind)  kind: journal | conf | book | preprint | report
REFS: dict[str, tuple[str, str | None, str]] = {
    # ---------------------------------------------------------------- gene prioritization and network medicine
    "barabasi2011": ("Barabási AL, Gulbahce N, Loscalzo J. Network medicine: a network-based approach to human disease. "
                     "Nature Reviews Genetics. 2011;12:56–68.", "10.1038/nrg2918", "journal"),
    "aerts2006": ("Aerts S, Lambrechts D, Maity S, et al. Gene prioritization through genomic data fusion. "
                  "Nature Biotechnology. 2006;24:537–544.", "10.1038/nbt1203", "journal"),
    "kohler2008": ("Köhler S, Bauer S, Horn D, Robinson PN. Walking the interactome for prioritization of candidate "
                   "disease genes. American Journal of Human Genetics. 2008;82:949–958.", "10.1016/j.ajhg.2008.02.013",
                   "journal"),
    "vanunu2010": ("Vanunu O, Magger O, Ruppin E, Shlomi T, Sharan R. Associating genes and protein complexes with "
                   "disease via network propagation. PLoS Computational Biology. 2010;6:e1000641.",
                   "10.1371/journal.pcbi.1000641", "journal"),
    "ghiassian2015": ("Ghiassian SD, Menche J, Barabási AL. A DIseAse MOdule Detection (DIAMOnD) algorithm derived from "
                      "a systematic analysis of connectivity patterns of disease proteins in the human interactome. "
                      "PLoS Computational Biology. 2015;11:e1004120.", "10.1371/journal.pcbi.1004120", "journal"),
    "kohler2009": ("Köhler S, Schulz MH, Krawitz P, et al. Clinical diagnostics in human genetics with semantic "
                   "similarity searches in ontologies. American Journal of Human Genetics. 2009;85:457–464.",
                   "10.1016/j.ajhg.2009.09.003", "journal"),
    "smedley2015": ("Smedley D, Jacobsen JOB, Jäger M, et al. Next-generation diagnostics and disease-gene discovery "
                    "with the Exomiser. Nature Protocols. 2015;10:2004–2015.", "10.1038/nprot.2015.124", "journal"),
    "pinero2020": ("Piñero J, Ramírez-Anguita JM, Saüch-Pitarch J, et al. The DisGeNET knowledge platform for disease "
                   "genomics: 2019 update. Nucleic Acids Research. 2020;48:D845–D855.", "10.1093/nar/gkz1021", "journal"),
    "wang2024kdgene": ("Wang X, Yang K, Jia T, et al. KDGene: knowledge graph completion for disease gene prediction "
                       "using interactional tensor decomposition. Briefings in Bioinformatics. 2024;25:bbae161.",
                       "10.1093/bib/bbae161", "journal"),
    "saarinen2024": ("Saarinen H, Goldsmith M, Wang RS, Loscalzo J, Maniscalco S. Disease gene prioritization with "
                     "quantum walks. Bioinformatics. 2024;40:btae513.", "10.1093/bioinformatics/btae513", "journal"),
    "frasca2020": ("Frasca F, Rossi E, Eynard D, Chamberlain B, Bronstein M, Monti F. SIGN: scalable inception graph "
                   "neural networks. arXiv:2004.11198. 2020.", None, "preprint"),
    "chandak2023": ("Chandak P, Huang K, Zitnik M. Building a knowledge graph to enable precision medicine. "
                    "Scientific Data. 2023;10:67.", "10.1038/s41597-023-01960-3", "journal"),
    "huang2024txgnn": ("Huang K, Chandak P, Wang Q, et al. A foundation model for clinician-centered drug repurposing. "
                       "Nature Medicine. 2024;30:3601–3613.", "10.1038/s41591-024-03233-x", "journal"),
    "gillis2012": ("Gillis J, Pavlidis P. “Guilt by association” is the exception rather than the rule in gene "
                   "networks. PLoS Computational Biology. 2012;8:e1002444.", "10.1371/journal.pcbi.1002444", "journal"),
    # ---------------------------------------------------------------- knowledge imbalance and annotation bias
    "edwards2011": ("Edwards AM, Isserlin R, Bader GD, Frye SV, Willson TM, Yu FH. Too many roads not taken. "
                    "Nature. 2011;470:163–165.", "10.1038/470163a", "journal"),
    "stoeger2018": ("Stoeger T, Gerlach M, Morimoto RI, Nunes Amaral LA. Large-scale investigation of the reasons why "
                    "potentially important genes are ignored. PLoS Biology. 2018;16:e2006643.",
                    "10.1371/journal.pbio.2006643", "journal"),
    "haynes2018": ("Haynes WA, Tomczak A, Khatri P. Gene annotation bias impedes biomedical research. Scientific "
                   "Reports. 2018;8:1362.", "10.1038/s41598-018-19333-x", "journal"),
    "schnoes2013": ("Schnoes AM, Ream DC, Thorman AW, Babbitt PC, Friedberg I. Biases in the experimental annotations "
                    "of protein function and their effect on our understanding of protein function space. PLoS "
                    "Computational Biology. 2013;9:e1003063.", "10.1371/journal.pcbi.1003063", "journal"),
    "kaufman2012": ("Kaufman S, Rosset S, Perlich C, Stitelman O. Leakage in data mining: formulation, detection, and "
                    "avoidance. ACM Transactions on Knowledge Discovery from Data. 2012;6:15.",
                    "10.1145/2382577.2382579", "journal"),
    "cawley2010": ("Cawley GC, Talbot NLC. On over-fitting in model selection and subsequent selection bias in "
                   "performance evaluation. Journal of Machine Learning Research. 2010;11:2079–2107.", None, "journal"),
    # ---------------------------------------------------------------- data resources
    "davis2025": ("Davis AP, Wiegers TC, Sciaky D, et al. Comparative Toxicogenomics Database's 20th anniversary: update 2025. "
                  "Nucleic Acids Research. 2025;53:D1328–D1334.", "10.1093/nar/gkae883", "journal"),
    "gargano2024": ("Gargano MA, Matentzoglu N, Coleman B, et al. The Human Phenotype Ontology in 2024: phenotypes "
                    "around the world. Nucleic Acids Research. 2024;52:D1333–D1346.", "10.1093/nar/gkad1005", "journal"),
    "ashburner2000": ("Ashburner M, Ball CA, Blake JA, et al. Gene Ontology: tool for the unification of biology. "
                      "Nature Genetics. 2000;25:25–29.", "10.1038/75556", "journal"),
    "go2023": ("The Gene Ontology Consortium. The Gene Ontology knowledgebase in 2023. Genetics. 2023;224:iyad031.",
               "10.1093/genetics/iyad031", "journal"),
    "szklarczyk2023": ("Szklarczyk D, Kirsch R, Koutrouli M, et al. The STRING database in 2023: protein–protein "
                       "association networks and functional enrichment analyses for any sequenced genome of interest. "
                       "Nucleic Acids Research. 2023;51:D638–D646.", "10.1093/nar/gkac1000", "journal"),
    "seal2023": ("Seal RL, Braschi B, Gray K, et al. Genenames.org: the HGNC resources in 2023. Nucleic Acids Research. "
                 "2023;51:D1003–D1009.", "10.1093/nar/gkac888", "journal"),
    "schriml2022": ("Schriml LM, Munro JB, Schor M, et al. The Human Disease Ontology 2022 update. Nucleic Acids "
                    "Research. 2022;50:D1255–D1261.", "10.1093/nar/gkab1063", "journal"),
    "amberger2019": ("Amberger JS, Bocchini CA, Scott AF, Hamosh A. OMIM.org: leveraging knowledge across "
                     "phenotype–gene relationships. Nucleic Acids Research. 2019;47:D1038–D1043.",
                     "10.1093/nar/gky1151", "journal"),
    "lipscomb2000": ("Lipscomb CE. Medical Subject Headings (MeSH). Bulletin of the Medical Library Association. "
                     "2000;88:265–266.", None, "journal"),
    "distefano2022": ("DiStefano MT, Goehringer S, Babb L, et al. The Gene Curation Coalition: a global effort to "
                      "harmonize gene–disease evidence resources. Genetics in Medicine. 2022;24:1732–1742.",
                      "10.1016/j.gim.2022.04.017", "journal"),
    "landrum2020": ("Landrum MJ, Chitipiralla S, Brown GR, et al. ClinVar: improvements to accessing data. Nucleic "
                    "Acids Research. 2020;48:D835–D844.", "10.1093/nar/gkz972", "journal"),
    "uhlen2015": ("Uhlén M, Fagerberg L, Hallström BM, et al. Tissue-based map of the human proteome. Science. "
                  "2015;347:1260419.", "10.1126/science.1260419", "journal"),
    "gtex2020": ("GTEx Consortium. The GTEx Consortium atlas of genetic regulatory effects across human tissues. "
                 "Science. 2020;369:1318–1330.", "10.1126/science.aaz1776", "journal"),
    "morales2022": ("Morales J, Pujar S, Loveland JE, et al. A joint NCBI and EMBL-EBI transcript set for clinical "
                    "genomics and research. Nature. 2022;604:310–315.", "10.1038/s41586-022-04558-8", "journal"),
    "uniprot2023": ("The UniProt Consortium. UniProt: the Universal Protein Knowledgebase in 2023. Nucleic Acids "
                    "Research. 2023;51:D523–D531.", "10.1093/nar/gkac1052", "journal"),
    "barrett2013": ("Barrett T, Wilhite SE, Ledoux P, et al. NCBI GEO: archive for functional genomics data sets—update. "
                    "Nucleic Acids Research. 2013;41:D991–D995.", "10.1093/nar/gks1193", "journal"),
    # ---------------------------------------------------------------- expression cohorts
    "gudjonsson2010": ("Gudjonsson JE, Ding J, Johnston A, et al. Assessment of the psoriatic transcriptome in a large "
                       "sample: additional regulated genes and comparisons with in vitro models. Journal of "
                       "Investigative Dermatology. 2010;130:1829–1840.", "10.1038/jid.2010.36", "journal"),
    "suarezfarinas2012": ("Suárez-Fariñas M, Li K, Fuentes-Duculan J, Hayden K, Brodmerkel C, Krueger JG. Expanding the "
                          "psoriasis disease profile: interrogation of the skin and serum of patients with "
                          "moderate-to-severe psoriasis. Journal of Investigative Dermatology. 2012;132:2552–2564.",
                          "10.1038/jid.2012.184", "journal"),
    "li2018uc": ("Li K, Strauss R, Ouahed J, et al. Molecular comparison of adult and pediatric ulcerative colitis "
                 "indicates broad similarity of molecular pathways in disease tissue. Journal of Pediatric "
                 "Gastroenterology and Nutrition. 2018;67:45–52.", "10.1097/MPG.0000000000001898", "journal"),
    "vancamelbeke2017": ("Vancamelbeke M, Vanuytsel T, Farré R, et al. Genetic and transcriptomic bases of intestinal "
                         "epithelial barrier dysfunction in inflammatory bowel disease. Inflammatory Bowel Diseases. "
                         "2017;23:1718–1729.", "10.1097/MIB.0000000000001246", "journal"),
    "ritchie2015": ("Ritchie ME, Phipson B, Wu D, et al. limma powers differential expression analyses for RNA-sequencing "
                    "and microarray studies. Nucleic Acids Research. 2015;43:e47.", "10.1093/nar/gkv007", "journal"),
    # ---------------------------------------------------------------- representation learning
    "liu2021sapbert": ("Liu F, Shareghi E, Meng Z, Basaldella M, Collier N. Self-alignment pretraining for biomedical "
                       "entity representations. In: Proceedings of NAACL-HLT. 2021:4228–4238.",
                       "10.18653/v1/2021.naacl-main.334", "conf"),
    "lee2020biobert": ("Lee J, Yoon W, Kim S, et al. BioBERT: a pre-trained biomedical language representation model "
                       "for biomedical text mining. Bioinformatics. 2020;36:1234–1240.",
                       "10.1093/bioinformatics/btz682", "journal"),
    "gu2021pubmedbert": ("Gu Y, Tinn R, Cheng H, et al. Domain-specific language model pretraining for biomedical natural "
                         "language processing. ACM Transactions on Computing for Healthcare. 2022;3:2.",
                         "10.1145/3458754", "journal"),
    "lin2023esm2": ("Lin Z, Akin H, Rao R, et al. Evolutionary-scale prediction of atomic-level protein structure with a "
                    "language model. Science. 2023;379:1123–1130.", "10.1126/science.ade2574", "journal"),
    "xu2023protst": ("Xu M, Yuan X, Miret S, Tang J. ProtST: multi-modality learning of protein sequences and biomedical "
                     "texts. In: Proceedings of the 40th International Conference on Machine Learning. PMLR 202. "
                     "2023:38749–38767.", None, "conf"),
    "radford2021clip": ("Radford A, Kim JW, Hallacy C, et al. Learning transferable visual models from natural language "
                        "supervision. In: Proceedings of the 38th International Conference on Machine Learning. PMLR 139. "
                        "2021:8748–8763.", None, "conf"),
    "hinton2015": ("Hinton G, Vinyals O, Dean J. Distilling the knowledge in a neural network. arXiv:1503.02531. 2015.",
                   None, "preprint"),
    "karpukhin2020": ("Karpukhin V, Oguz B, Min S, et al. Dense passage retrieval for open-domain question answering. "
                      "In: Proceedings of EMNLP. 2020:6769–6781.", "10.18653/v1/2020.emnlp-main.550", "conf"),
    "nogueira2019": ("Nogueira R, Cho K. Passage re-ranking with BERT. arXiv:1901.04085. 2019.", None, "preprint"),
    # ---------------------------------------------------------------- ranking, fusion, learning
    "cormack2009": ("Cormack GV, Clarke CLA, Büttcher S. Reciprocal rank fusion outperforms Condorcet and individual "
                    "rank learning methods. In: Proceedings of SIGIR. 2009:758–759.", "10.1145/1571941.1572114", "conf"),
    "ke2017": ("Ke G, Meng Q, Finley T, et al. LightGBM: a highly efficient gradient boosting decision tree. In: Advances "
               "in Neural Information Processing Systems 30. 2017:3146–3154.", None, "conf"),
    "burges2010": ("Burges CJC. From RankNet to LambdaRank to LambdaMART: an overview. Microsoft Research Technical "
                   "Report MSR-TR-2010-82. 2010.", None, "report"),
    "elkan2008": ("Elkan C, Noto K. Learning classifiers from only positive and unlabeled data. In: Proceedings of KDD. "
                  "2008:213–220.", "10.1145/1401890.1401920", "conf"),
    "bekker2020": ("Bekker J, Davis J. Learning from positive and unlabeled data: a survey. Machine Learning. "
                   "2020;109:719–760.", "10.1007/s10994-020-05877-5", "journal"),
    "loshchilov2019": ("Loshchilov I, Hutter F. Decoupled weight decay regularization. In: International Conference on "
                       "Learning Representations. 2019.", None, "conf"),
    # ---------------------------------------------------------------- statistics, explanation, visualisation
    "efron1993": ("Efron B, Tibshirani RJ. An Introduction to the Bootstrap. New York: Chapman & Hall; 1993.", None, "book"),
    "field2007": ("Field CA, Welsh AH. Bootstrapping clustered data. Journal of the Royal Statistical Society: Series B. "
                  "2007;69:369–390.", "10.1111/j.1467-9868.2007.00593.x", "journal"),
    "lundberg2017": ("Lundberg SM, Lee SI. A unified approach to interpreting model predictions. In: Advances in Neural "
                     "Information Processing Systems 30. 2017:4765–4774.", None, "conf"),
    "lundberg2020": ("Lundberg SM, Erion G, Chen H, et al. From local explanations to global understanding with "
                     "explainable AI for trees. Nature Machine Intelligence. 2020;2:56–67.",
                     "10.1038/s42256-019-0138-9", "journal"),
    "vandermaaten2008": ("van der Maaten L, Hinton G. Visualizing data using t-SNE. Journal of Machine Learning Research. "
                         "2008;9:2579–2605.", None, "journal"),
    "lex2014": ("Lex A, Gehlenborg N, Strobelt H, Vuillemot R, Pfister H. UpSet: visualization of intersecting sets. IEEE "
                "Transactions on Visualization and Computer Graphics. 2014;20:1983–1992.",
                "10.1109/TVCG.2014.2346248", "journal"),
    "machado2009": ("Machado GM, Oliveira MM, Fernandes LAF. A physiologically-based model for simulation of color vision "
                    "deficiency. IEEE Transactions on Visualization and Computer Graphics. 2009;15:1291–1298.",
                    "10.1109/TVCG.2009.113", "journal"),
}


# ------------------------------------------------------------------ Crossref verification
def _get(url: str) -> dict | None:
    req = urllib.request.Request(url, headers={"User-Agent": "sara-manuscript-reference-check/1.0"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception:
            time.sleep(1.5 * (attempt + 1))
    return None


def _norm(s: str) -> str:
    s = re.sub(r"<[^>]+>", "", s or "")
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def _title_of(text: str) -> str:
    parts = text.split(". ")
    return parts[1] if len(parts) > 2 else text


def _year_of(text: str) -> int | None:
    m = re.search(r"\. ((?:19|20)\d{2})[;:.]", text) or re.search(r"[;,] ((?:19|20)\d{2})\.", text)
    return int(m.group(1)) if m else None


# classic references without a DOI whose details were checked against the publisher pages by hand
MANUAL_OK = {"cawley2010", "lipscomb2000", "vandermaaten2008"}


def verify(keys: list[str] | None = None) -> dict:
    cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
    out = {}
    for key in keys or list(REFS):
        text, doi, kind = REFS[key]
        rec = {"kind": kind, "doi": doi}
        if doi:
            if doi not in cache:
                cache[doi] = _get("https://api.crossref.org/works/" + urllib.parse.quote(doi))
                time.sleep(0.4)
            msg = (cache.get(doi) or {}).get("message")
            if not msg:
                rec.update(status="doi_not_found")
            else:
                ct = " ".join([(msg.get("title") or [""])[0]] + list(msg.get("subtitle") or []))
                sim = difflib.SequenceMatcher(None, _norm(ct), _norm(_title_of(text))).ratio()
                fa = (msg.get("author") or [{}])[0].get("family", "")
                yr = None
                for k in ("published-print", "published-online", "issued"):
                    dp = (msg.get(k) or {}).get("date-parts")
                    if dp and dp[0] and dp[0][0]:
                        yr = dp[0][0]
                        break
                ty = _year_of(text)
                ok_author = _norm(fa).split(" ")[0] in _norm(text.split(",")[0]) if fa else ("Consortium" in text)
                rec.update(crossref_title=ct, title_similarity=round(sim, 3), first_author=fa, year=yr,
                           journal=(msg.get("container-title") or [""])[0],
                           status="ok" if sim >= 0.85 and (yr is None or ty is None or abs(yr - ty) <= 1) and ok_author
                           else "mismatch")
        else:
            q = _title_of(text)
            ck = "q:" + q
            if ck not in cache:
                cache[ck] = _get("https://api.crossref.org/works?rows=3&query.bibliographic=" + urllib.parse.quote(q))
                time.sleep(0.4)
            items = ((cache.get(ck) or {}).get("message") or {}).get("items") or []
            scored = [(difflib.SequenceMatcher(None, _norm((it.get("title") or [""])[0]), _norm(q)).ratio(), i)
                      for i, it in enumerate(items)]
            best = (max(scored)[0], items[max(scored)[1]]) if scored else (0.0, {})
            rec.update(query_title=q, best_similarity=round(best[0], 3),
                       best_title=(best[1].get("title") or [""])[0] if best[1] else "",
                       status="ok_noid" if best[0] >= 0.85 or kind in ("preprint", "book", "report", "conf") else "unverified")
        if key in MANUAL_OK and rec.get("status") not in ("ok", "ok_noid"):
            rec["status"] = "manual_ok"
        out[key] = rec
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    return out


# ------------------------------------------------------------------ citation resolution
CITE_RX = re.compile(r"\[@([^\]]+)\]")


def _fmt_nums(nums: list[int]) -> str:
    nums = sorted(set(nums))
    out, i = [], 0
    while i < len(nums):
        j = i
        while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
            j += 1
        out.append(f"{nums[i]}–{nums[j]}" if j - i >= 2 else ", ".join(str(n) for n in nums[i:j + 1]))
        i = j + 1
    return "[" + ", ".join(out) + "]"


def resolve(md: str, order: dict | None = None) -> tuple[str, list[str], dict]:
    """Number references by first appearance; returns (text, reference list, key->number)."""
    order = dict(order or {})

    def rep(m):
        keys = [k.strip().lstrip("@") for k in m.group(1).split(";")]
        nums = []
        for k in keys:
            if k not in REFS:
                raise KeyError(f"unknown reference key: {k}")
            if k not in order:
                order[k] = len(order) + 1
            nums.append(order[k])
        return _fmt_nums(nums)

    text = CITE_RX.sub(rep, md)
    lst = [f"{n}. {REFS[k][0]}" + (f" doi:{REFS[k][1]}" if REFS[k][1] else "")
           for k, n in sorted(order.items(), key=lambda t: t[1])]
    return text, lst, order


if __name__ == "__main__":
    rep = verify()
    bad = {k: v for k, v in rep.items() if v["status"] not in ("ok", "ok_noid", "manual_ok")}
    print(json.dumps(bad, ensure_ascii=False, indent=1))
    print(len(rep), "checked;", len(bad), "need attention")
    (CACHE.parent / "refs_verification.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")

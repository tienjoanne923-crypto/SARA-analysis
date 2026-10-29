"""References of the English SARA manuscript in the Briefings in Bioinformatics (OUP) style.

* In-text citations are bracketed numbers in order of first citation: [1], [2, 5], [1–4] (a range for three or more
  consecutive numbers).
* Entries: 'Köhler S, Bauer S, Horn D, et al. Title. Am J Hum Genet 2008;82:949–58.' -- up to three authors then
  et al., NLM journal abbreviations, volume without issue, abbreviated page ranges; proceedings, arXiv preprints,
  books and reports in the corresponding OUP forms.
* Bibliographic data: the Crossref records cached for the Chinese package (refs_cache.json; authors in their
  published case) and the curated titles/years of refs9; references without a DOI are curated below.
"""
from __future__ import annotations

import json
import re

from .refs9 import REFS
from .refs10 import CACHE

CITE_RX = re.compile(r"\[@([^\]]+)\]")
JOURNAL = {
    "Nature Reviews Genetics": "Nat Rev Genet", "The American Journal of Human Genetics": "Am J Hum Genet",
    "PLoS Computational Biology": "PLoS Comput Biol", "PLOS Computational Biology": "PLoS Comput Biol",
    "Nature Biotechnology": "Nat Biotechnol", "Nature Protocols": "Nat Protoc", "Scientific Data": "Sci Data",
    "Nature Medicine": "Nat Med", "Briefings in Bioinformatics": "Brief Bioinform", "Bioinformatics": "Bioinformatics",
    "Nature": "Nature", "PLOS Biology": "PLoS Biol", "Scientific Reports": "Sci Rep",
    "ACM Transactions on Knowledge Discovery from Data": "ACM Trans Knowl Discov Data",
    "ACM Transactions on Computing for Healthcare": "ACM Trans Comput Healthc", "Science": "Science",
    "Nucleic Acids Research": "Nucleic Acids Res", "Nature Genetics": "Nat Genet", "GENETICS": "Genetics",
    "Genetics": "Genetics", "Machine Learning": "Mach Learn", "Genetics in Medicine": "Genet Med",
    "Journal of the Royal Statistical Society Series B: Statistical Methodology": "J R Stat Soc Series B Stat Methodol",
    "Nature Machine Intelligence": "Nat Mach Intell",
    "IEEE Transactions on Visualization and Computer Graphics": "IEEE Trans Vis Comput Graph",
    "Journal of Investigative Dermatology": "J Invest Dermatol",
    "Journal of Pediatric Gastroenterology and Nutrition": "J Pediatr Gastroenterol Nutr",
    "Inflammatory Bowel Diseases": "Inflamm Bowel Dis", "Journal of Machine Learning Research": "J Mach Learn Res",
    "Bulletin of the Medical Library Association": "Bull Med Libr Assoc",
}
CONSORTIUM = {"The Gene Ontology Consortium": "Gene Ontology Consortium", "The UniProt Consortium": "UniProt Consortium",
              "The GTEx Consortium": "GTEx Consortium", "GTEx Consortium": "GTEx Consortium"}
PROC_PLACE = {"Association for Computational Linguistics": "Stroudsburg, PA", "ACM": "New York, NY"}
PROC_FIX = {
    "Proceedings of the 32nd international ACM SIGIR conference on Research and development in information retrieval":
        "Proceedings of the 32nd International ACM SIGIR Conference on Research and Development in Information Retrieval",
    "Proceedings of the 14th ACM SIGKDD international conference on Knowledge discovery and data mining":
        "Proceedings of the 14th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining"}
MANUAL = {
    "frasca2020": {"type": "arxiv", "authors": ["Frasca F", "Rossi E", "Eynard D", "et al."],
                   "title": "SIGN: scalable inception graph neural networks", "year": 2020, "arxiv": "2004.11198"},
    "cawley2010": {"type": "J", "authors": ["Cawley GC", "Talbot NLC"],
                   "title": "On over-fitting in model selection and subsequent selection bias in performance evaluation",
                   "journal": "J Mach Learn Res", "year": 2010, "volume": "11", "pages": "2079-2107"},
    "lipscomb2000": {"type": "J", "authors": ["Lipscomb CE"], "title": "Medical Subject Headings (MeSH)",
                     "journal": "Bull Med Libr Assoc", "year": 2000, "volume": "88", "pages": "265-266"},
    "xu2023protst": {"type": "C", "authors": ["Xu M", "Yuan X", "Miret S", "et al."],
                     "title": "ProtST: multi-modality learning of protein sequences and biomedical texts",
                     "proceedings": "Proceedings of the 40th International Conference on Machine Learning (PMLR 202)",
                     "publisher": "PMLR", "year": 2023, "pages": "38749-38767"},
    "radford2021clip": {"type": "C", "authors": ["Radford A", "Kim JW", "Hallacy C", "et al."],
                        "title": "Learning transferable visual models from natural language supervision",
                        "proceedings": "Proceedings of the 38th International Conference on Machine Learning (PMLR 139)",
                        "publisher": "PMLR", "year": 2021, "pages": "8748-8763"},
    "hinton2015": {"type": "arxiv", "authors": ["Hinton G", "Vinyals O", "Dean J"],
                   "title": "Distilling the knowledge in a neural network", "year": 2015, "arxiv": "1503.02531"},
    "nogueira2019": {"type": "arxiv", "authors": ["Nogueira R", "Cho K"], "title": "Passage re-ranking with BERT",
                     "year": 2019, "arxiv": "1901.04085"},
    "ke2017": {"type": "C", "authors": ["Ke G", "Meng Q", "Finley T", "et al."],
               "title": "LightGBM: a highly efficient gradient boosting decision tree",
               "proceedings": "Advances in Neural Information Processing Systems 30", "place": "Red Hook, NY",
               "publisher": "Curran Associates", "year": 2017, "pages": "3146-3154"},
    "burges2010": {"type": "R", "authors": ["Burges CJC"],
                   "title": "From RankNet to LambdaRank to LambdaMART: an overview", "report": "Technical Report MSR-TR-2010-82",
                   "place": "Redmond, WA", "publisher": "Microsoft Research", "year": 2010},
    "loshchilov2019": {"type": "C", "authors": ["Loshchilov I", "Hutter F"], "title": "Decoupled weight decay regularization",
                       "proceedings": "Proceedings of the 7th International Conference on Learning Representations (ICLR)",
                       "place": "New Orleans, LA", "year": 2019},
    "efron1993": {"type": "M", "authors": ["Efron B", "Tibshirani RJ"], "title": "An Introduction to the Bootstrap",
                  "place": "New York, NY", "publisher": "Chapman & Hall", "year": 1993},
    "lundberg2017": {"type": "C", "authors": ["Lundberg SM", "Lee SI"],
                     "title": "A unified approach to interpreting model predictions",
                     "proceedings": "Advances in Neural Information Processing Systems 30", "place": "Red Hook, NY",
                     "publisher": "Curran Associates", "year": 2017, "pages": "4765-4774"},
    "vandermaaten2008": {"type": "J", "authors": ["van der Maaten L", "Hinton G"], "title": "Visualizing data using t-SNE",
                         "journal": "J Mach Learn Res", "year": 2008, "volume": "9", "pages": "2579-2605"},
}
OVERRIDE = {"pinero2020": {"volume": "48", "pages": "D845-D855", "year": 2020},
            "barrett2013": {"year": 2013},
            "kaufman2012": {"title": "Leakage in data mining: formulation, detection, and avoidance"},
            "cormack2009": {"authors": ["Cormack GV", "Clarke CLA", "Büttcher S"]},
            "machado2009": {"authors": ["Machado GM", "Oliveira MM", "Fernandes LAF"]}}
PARTICLES = {"van", "der", "de", "den", "von", "da", "di", "le", "la"}


def _initials(given: str) -> str:
    return "".join(p[0].upper() for p in re.split(r"[\s\-\.]+", given or "") if p)


def _family(f: str) -> str:
    if f.isupper() and len(f) > 3:                                   # the rare all-capitals Crossref record
        f = " ".join(w if w.lower() in PARTICLES else w.capitalize() for w in f.lower().split())
    return f


def _authors(m: dict) -> list[str]:
    out = []
    for a in m.get("author", []):
        if a.get("family"):
            out.append(f"{_family(a['family'])} {_initials(a.get('given', ''))}".strip())
        elif a.get("name"):
            out.append(CONSORTIUM.get(a["name"], a["name"]))
    return out


def _title_refs9(text: str) -> str:
    m = re.match(r"^(.*?(?:et al|[A-Z]{1,4}|Consortium))\. (.+?)\. ", text)
    return m.group(2) if m else ""


def _year_refs9(text: str) -> int | None:
    m = re.search(r"\b(19|20)\d{2}\b(?=;|\.|:)", text)
    return int(m.group(0)) if m else None


def record(key: str, cache: dict) -> dict:
    text, doi, _ = REFS[key]
    if key in MANUAL:
        return {**MANUAL[key], "key": key, "doi": doi}
    m = cache[doi]["message"]
    r = {"key": key, "doi": doi, "authors": _authors(m),
         "title": _title_refs9(text) or re.sub(r"<[^>]+>", "", (m.get("title") or [""])[0]),
         "year": _year_refs9(text) or m["issued"]["date-parts"][0][0]}
    if m.get("type") == "proceedings-article":
        pub = m.get("publisher", "")
        proc = (m.get("container-title") or [""])[0]
        r.update(type="C", proceedings=PROC_FIX.get(proc, proc), publisher=pub,
                 place=PROC_PLACE.get(pub, ""), pages=(m.get("page") or "").replace("–", "-"))
    else:
        j = (m.get("container-title") or [""])[0]
        r.update(type="J", journal=JOURNAL[j], volume=m.get("volume"),
                 pages=(m.get("page") or m.get("article-number") or "").replace("–", "-"))
        if r["pages"] and re.fullmatch(r"(e\d+)-\1", r["pages"]):
            r["pages"] = r["pages"].split("-")[0]
    r.update(OVERRIDE.get(key, {}))
    return r


def pages(p: str | None) -> str:
    """NLM page range: 949-958 -> 949–58, D1003-D1009 -> D1003–9, single pages and article numbers unchanged."""
    if not p:
        return ""
    if "-" not in p:
        return p
    a, b = p.split("-", 1)
    ma, mb = re.fullmatch(r"([A-Za-z]*)(\d+)", a), re.fullmatch(r"([A-Za-z]*)(\d+)", b)
    if not (ma and mb) or ma.group(1) != mb.group(1):
        return f"{a}–{b}"
    da, db = ma.group(2), mb.group(2)
    if len(da) == len(db):
        k = 0
        while k < len(db) - 1 and da[k] == db[k]:
            k += 1
        db = db[k:]
    return f"{a}–{db}"


def authors_field(names: list[str]) -> str:
    etal = "et al." in names
    names = [n for n in names if n != "et al."]
    if len(names) > 3:
        names, etal = names[:3], True
    return ", ".join(names) + (", et al." if etal else ".")


def bib(r: dict) -> str:
    """one reference entry (without its number)."""
    A, T, y = authors_field(r["authors"]), r["title"].rstrip("."), r["year"]
    t = r["type"]
    if t == "J":
        pg = pages(r.get("pages"))
        return f"{A} {T}. {r['journal']} {y};{r['volume']}{':' + pg if pg else ''}."
    if t == "C":
        where = ". ".join(x for x in (r.get("place", "") + (": " if r.get("place") and r.get("publisher") else "") +
                                      r.get("publisher", ""),) if x)
        pg = pages(r.get("pages"))
        tail = ", ".join(x for x in (where, str(y), pg) if x)
        return f"{A} {T}. In: {r['proceedings']}. {tail}."
    if t == "arxiv":
        return f"{A} {T}. arXiv preprint arXiv:{r['arxiv']}, {y}."
    if t == "M":
        return f"{A} {T}. {r['place']}: {r['publisher']}, {y}."
    if t == "R":
        return f"{A} {T}. {r['report']}. {r['place']}: {r['publisher']}, {y}."
    raise ValueError(t)


def nums(ns: list[int]) -> str:
    ns = sorted(set(ns))
    out, i = [], 0
    while i < len(ns):
        j = i
        while j + 1 < len(ns) and ns[j + 1] == ns[j] + 1:
            j += 1
        out.append(f"{ns[i]}–{ns[j]}" if j - i >= 2 else ", ".join(str(n) for n in ns[i:j + 1]))
        i = j + 1
    return "[" + ", ".join(out) + "]"


def resolve(md: str, order: dict | None = None) -> tuple[str, list[str], dict, list[dict]]:
    """[@k1; @k2] -> [1, 2]; returns (text, entries 'n. ...', key->n, records)."""
    cache = json.loads(CACHE.read_text(encoding="utf-8"))
    order = dict(order or {})

    def rep(m):
        ns = []
        for k in [x.strip().lstrip("@") for x in m.group(1).split(";")]:
            if k not in REFS:
                raise KeyError(k)
            order.setdefault(k, len(order) + 1)
            ns.append(order[k])
        return nums(ns)
    text = CITE_RX.sub(rep, md)
    recs = [record(k, cache) for k, _ in sorted(order.items(), key=lambda t: t[1])]
    entries = [f"{order[r['key']]}. {bib(r)}" for r in recs]
    return text, entries, order, recs


def ris(recs: list[dict]) -> str:
    ty = {"J": "JOUR", "C": "CONF", "M": "BOOK", "R": "RPRT", "arxiv": "ELEC"}
    out = []
    for r in recs:
        lines = [f"TY  - {ty[r['type']]}"]
        lines += [f"AU  - {a}" for a in r["authors"] if a != "et al."]
        lines += [f"TI  - {r['title']}", f"PY  - {r['year']}"]
        if r.get("journal"):
            lines.append(f"JO  - {r['journal']}")
        if r.get("proceedings"):
            lines.append(f"T2  - {r['proceedings']}")
        for k, tag in (("volume", "VL"), ("publisher", "PB"), ("place", "CY"), ("doi", "DO")):
            if r.get(k):
                lines.append(f"{tag}  - {r[k]}")
        if r.get("arxiv"):
            lines.append(f"UR  - https://arxiv.org/abs/{r['arxiv']}")
        if r.get("pages"):
            sp = r["pages"].split("-")
            lines.append(f"SP  - {sp[0]}")
            if len(sp) > 1:
                lines.append(f"EP  - {sp[1]}")
        lines.append("ER  - ")
        out.append("\n".join(lines))
    return "\n\n".join(out) + "\n"


def check(recs: list[dict]) -> list[str]:
    bad = []
    for r in recs:
        e = bib(r)
        if r["type"] == "J" and not (r.get("volume") and r.get("pages")):
            bad.append(f"{r['key']}: journal entry lacks volume/pages")
        if re.search(r"[A-Z]{3,} [A-Z]", e.split(".")[0]) and "Consortium" not in e:
            bad.append(f"{r['key']}: author names in capitals")
        if not re.match(r"^[A-Za-zÀ-ÿ].*\.$", e):
            bad.append(f"{r['key']}: malformed entry")
    return bad


if __name__ == "__main__":
    cache = json.loads(CACHE.read_text(encoding="utf-8"))
    recs = [record(k, cache) for k in REFS]
    for r in recs:
        print(bib(r))
    print("problems:", check(recs))

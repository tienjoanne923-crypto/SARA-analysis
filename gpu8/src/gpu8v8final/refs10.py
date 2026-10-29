"""References in GB/T 7714—2015 (sequential numbering) for the SARA manuscript.

* DOI references: authors, container, volume, issue and pages from the Crossref records cached by refs9
  (submission9/refs_cache.json); titles and years from the curated refs9 entries (sentence case, print year).
* References without a DOI: curated records checked against the official pages (PMLR, JMLR, arXiv, NeurIPS,
  PubMed Central) on 2026-09-30.
* In-text citations become superscript "[1-4]" / "[2,5]" markers (<sup>[...]</sup> in the Markdown), numbered by
  first appearance; the builder turns them into superscript hyperlinks to the reference entries.
"""
from __future__ import annotations

import difflib
import json
import re
from pathlib import Path

from .refs9 import REFS

CACHE = Path(r"D:\信\gpu8\submission9\refs_cache.json")
ACCESS = "2026-09-30"
CITE_RX = re.compile(r"\[@([^\]]+)\]")

# curated records (no DOI, or Crossref fields to override)
MANUAL = {
    "frasca2020": {"type": "EB/OL", "authors": ["FRASCA F", "ROSSI E", "EYNARD D", "CHAMBERLAIN B", "BRONSTEIN M", "MONTI F"],
                   "title": "SIGN: scalable inception graph neural networks", "year": 2020,
                   "url": "https://arxiv.org/abs/2004.11198"},
    "cawley2010": {"type": "J", "authors": ["CAWLEY G C", "TALBOT N L C"],
                   "title": "On over-fitting in model selection and subsequent selection bias in performance evaluation",
                   "journal": "Journal of Machine Learning Research", "year": 2010, "volume": "11", "issue": "70",
                   "pages": "2079-2107"},
    "lipscomb2000": {"type": "J", "authors": ["LIPSCOMB C E"], "title": "Medical Subject Headings (MeSH)",
                     "journal": "Bulletin of the Medical Library Association", "year": 2000, "volume": "88", "issue": "3",
                     "pages": "265-266"},
    "xu2023protst": {"type": "C", "authors": ["XU M", "YUAN X", "MIRET S", "TANG J"],
                     "title": "ProtST: multi-modality learning of protein sequences and biomedical texts",
                     "proceedings": "Proceedings of the 40th International Conference on Machine Learning: Volume 202",
                     "place": "[S.l.]", "publisher": "PMLR", "year": 2023, "pages": "38749-38767"},
    "radford2021clip": {"type": "C", "authors": ["RADFORD A", "KIM J W", "HALLACY C", "et al."],
                        "title": "Learning transferable visual models from natural language supervision",
                        "proceedings": "Proceedings of the 38th International Conference on Machine Learning: Volume 139",
                        "place": "[S.l.]", "publisher": "PMLR", "year": 2021, "pages": "8748-8763"},
    "hinton2015": {"type": "EB/OL", "authors": ["HINTON G", "VINYALS O", "DEAN J"],
                   "title": "Distilling the knowledge in a neural network", "year": 2015, "date": "2015-03-09",
                   "url": "https://arxiv.org/abs/1503.02531"},
    "nogueira2019": {"type": "EB/OL", "authors": ["NOGUEIRA R", "CHO K"], "title": "Passage re-ranking with BERT",
                     "year": 2019, "date": "2019-01-13", "url": "https://arxiv.org/abs/1901.04085"},
    "ke2017": {"type": "C/OL", "authors": ["KE G", "MENG Q", "FINLEY T", "et al."],
               "title": "LightGBM: a highly efficient gradient boosting decision tree",
               "proceedings": "Advances in Neural Information Processing Systems 30", "place": "Red Hook",
               "publisher": "Curran Associates", "year": 2017,
               "url": "https://proceedings.neurips.cc/paper_files/paper/2017/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf"},
    "burges2010": {"type": "R", "authors": ["BURGES C J C"],
                   "title": "From RankNet to LambdaRank to LambdaMART: an overview: MSR-TR-2010-82",
                   "place": "Redmond", "publisher": "Microsoft Research", "year": 2010},
    "loshchilov2019": {"type": "C/OL", "authors": ["LOSHCHILOV I", "HUTTER F"], "title": "Decoupled weight decay regularization",
                       "proceedings": "7th International Conference on Learning Representations", "place": "New Orleans",
                       "publisher": "ICLR", "year": 2019, "url": "https://arxiv.org/abs/1711.05101"},
    "efron1993": {"type": "M", "authors": ["EFRON B", "TIBSHIRANI R J"], "title": "An introduction to the bootstrap",
                  "place": "New York", "publisher": "Chapman & Hall", "year": 1993},
    "lundberg2017": {"type": "C/OL", "authors": ["LUNDBERG S M", "LEE S I"],
                     "title": "A unified approach to interpreting model predictions",
                     "proceedings": "Advances in Neural Information Processing Systems 30", "place": "Red Hook",
                     "publisher": "Curran Associates", "year": 2017,
                     "url": "https://proceedings.neurips.cc/paper_files/paper/2017/file/8a20a8621978632d76c43dfd28b67767-Paper.pdf"},
    "vandermaaten2008": {"type": "J", "authors": ["VAN DER MAATEN L", "HINTON G"], "title": "Visualizing data using t-SNE",
                         "journal": "Journal of Machine Learning Research", "year": 2008, "volume": "9", "issue": "86",
                         "pages": "2579-2605"},
}
OVERRIDE = {"pinero2020": {"volume": "48", "issue": "D1", "pages": "D845-D855", "year": 2020},
            "barrett2013": {"year": 2013},
            "kaufman2012": {"title": "Leakage in data mining: formulation, detection, and avoidance"},
            "cormack2009": {"authors": ["CORMACK G V", "CLARKE C L A", "BÜTTCHER S"],
                            "author_field": "CORMACK G V, CLARKE C L A, BÜTTCHER S"},
            "machado2009": {"authors": ["MACHADO G M", "OLIVEIRA M M", "FERNANDES L A F"],
                            "author_field": "MACHADO G M, OLIVEIRA M M, FERNANDES L A F"}}
PLACE = {"Association for Computational Linguistics": "Stroudsburg", "ACM": "New York"}
JOURNAL_FIX = {"GENETICS": "Genetics"}


def _up(s: str) -> str:
    return s.upper()


def _initials(given: str) -> str:
    parts = [p for p in re.split(r"[\s\-\.]+", given or "") if p]
    return " ".join(p[0].upper() for p in parts)


def _authors_crossref(m: dict) -> list[str]:
    out = []
    for a in m.get("author", []):
        if a.get("family"):
            out.append((_up(a["family"]) + " " + _initials(a.get("given", ""))).strip())
        elif a.get("name"):
            out.append(a["name"])
    return out


def _author_field(names: list[str]) -> str:
    names = [n for n in names if n != "et al."] if "et al." in names else names
    if len(names) > 3 or "et al." in MANUAL.get("", {}):
        return ", ".join(names[:3]) + ", et al"
    return ", ".join(names)


def _title_from_refs9(text: str) -> str:
    """'Authors. Title. Journal...' -> Title (the second sentence of the curated entry)."""
    m = re.match(r"^(.*?(?:et al|[A-Z]{1,4}|Consortium))\. (.+?)\. ", text)
    return m.group(2) if m else ""


def _year_from_refs9(text: str) -> int | None:
    m = re.search(r"\b(19|20)\d{2}\b(?=;|\.|:)", text)
    return int(m.group(0)) if m else None


def _cap(s: str) -> str:
    return s[:1].upper() + s[1:] if s else s


def record(key: str, cache: dict) -> dict:
    text, doi, kind = REFS[key]
    if key in MANUAL:
        r = dict(MANUAL[key])
        au = r["authors"]
        r["author_field"] = (", ".join([a for a in au if a != "et al."][:3]) + (", et al" if "et al." in au or len(au) > 3 else ""))
        r["key"], r["doi"] = key, doi
        return r
    m = cache[doi]["message"]
    au = _authors_crossref(m)
    r = {"key": key, "doi": doi, "authors": au,
         "author_field": ", ".join(au[:3]) + (", et al" if len(au) > 3 else ""),
         "title": _title_from_refs9(text) or re.sub(r"<[^>]+>", "", (m.get("title") or [""])[0]),
         "year": _year_from_refs9(text) or m["issued"]["date-parts"][0][0]}
    if m.get("type") == "proceedings-article":
        pub = m.get("publisher", "")
        r.update(type="C", proceedings=(m.get("container-title") or [""])[0], publisher=pub,
                 place=PLACE.get(pub, "[S.l.]"), pages=(m.get("page") or "").replace("–", "-"))
    else:
        j = (m.get("container-title") or [""])[0]
        r.update(type="J", journal=JOURNAL_FIX.get(j, j), volume=m.get("volume"), issue=m.get("issue"),
                 pages=(m.get("page") or m.get("article-number") or "").replace("–", "-"))
        if r["pages"] and re.fullmatch(r"(e\d+)-\1", r["pages"]):
            r["pages"] = r["pages"].split("-")[0]
    r.update(OVERRIDE.get(key, {}))
    return r


def gbt(r: dict) -> str:
    """one GB/T 7714—2015 entry (without the [n] number)."""
    A, T, y = r["author_field"], r["title"].rstrip("."), r["year"]
    doi = f" DOI:{r['doi']}." if r.get("doi") else ""
    t = r["type"]
    if t == "J":
        vi = f"{r['volume']}" + (f"({r['issue']})" if r.get("issue") else "")
        pg = f": {r['pages']}" if r.get("pages") else ""
        return f"{A}. {T}[J]. {r['journal']}, {y}, {vi}{pg}.{doi}"
    if t == "C":
        return f"{A}. {T}[C]//{r['proceedings']}. {r['place']}: {r['publisher']}, {y}: {r['pages']}.{doi}"
    if t == "C/OL":
        return f"{A}. {T}[C/OL]//{r['proceedings']}. {r['place']}: {r['publisher']}, {y}[{ACCESS}]. {r['url']}."
    if t == "M":
        return f"{A}. {T}[M]. {r['place']}: {r['publisher']}, {y}."
    if t == "R":
        return f"{A}. {T}[R]. {r['place']}: {r['publisher']}, {y}."
    if t == "EB/OL":
        d = r.get("date", str(y))
        return f"{A}. {T}[EB/OL]. ({d})[{ACCESS}]. {r['url']}."
    raise ValueError(t)


def _nums(nums: list[int]) -> str:
    nums = sorted(set(nums))
    out, i = [], 0
    while i < len(nums):
        j = i
        while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
            j += 1
        out.append(f"{nums[i]}-{nums[j]}" if j - i >= 2 else ",".join(str(n) for n in nums[i:j + 1]))
        i = j + 1
    return "[" + ",".join(out) + "]"


def resolve(md: str, order: dict | None = None) -> tuple[str, list[str], dict, list[dict]]:
    """[@k1; @k2] -> <sup>[1,2]</sup>; returns (text, entries '[n] ...', key->n, records)."""
    cache = json.loads(CACHE.read_text(encoding="utf-8"))
    order = dict(order or {})

    def rep(m):
        nums = []
        for k in [x.strip().lstrip("@") for x in m.group(1).split(";")]:
            if k not in REFS:
                raise KeyError(k)
            order.setdefault(k, len(order) + 1)
            nums.append(order[k])
        return f"<sup>{_nums(nums)}</sup>"
    text = CITE_RX.sub(rep, md)
    recs = [record(k, cache) for k, _ in sorted(order.items(), key=lambda t: t[1])]
    entries = [f"[{order[r['key']]}] {gbt(r)}" for r in recs]
    return text, entries, order, recs


def ris(recs: list[dict]) -> str:
    ty = {"J": "JOUR", "C": "CONF", "C/OL": "CONF", "M": "BOOK", "R": "RPRT", "EB/OL": "ELEC"}
    out = []
    for r in recs:
        lines = [f"TY  - {ty[r['type']]}"]
        for a in r["authors"]:
            if a != "et al.":
                lines.append(f"AU  - {a}")
        lines += [f"TI  - {_cap(r['title'])}", f"PY  - {r['year']}"]
        if r.get("journal"):
            lines.append(f"JO  - {r['journal']}")
        if r.get("proceedings"):
            lines.append(f"T2  - {r['proceedings']}")
        for k, tag in (("volume", "VL"), ("issue", "IS"), ("publisher", "PB"), ("place", "CY"), ("url", "UR"), ("doi", "DO")):
            if r.get(k):
                lines.append(f"{tag}  - {r[k]}")
        if r.get("pages"):
            sp = r["pages"].split("-")
            lines.append(f"SP  - {sp[0]}")
            if len(sp) > 1:
                lines.append(f"EP  - {sp[1]}")
        lines.append("ER  - ")
        out.append("\n".join(lines))
    return "\n\n".join(out) + "\n"


def check(recs: list[dict]) -> list[str]:
    """format checks used by verify10."""
    cache = json.loads(CACHE.read_text(encoding="utf-8"))
    bad = []
    for r in recs:
        e = gbt(r)
        if r["type"] == "J" and not (r.get("volume") and r.get("pages")):
            bad.append(f"{r['key']}: journal entry lacks volume/pages")
        if re.search(r"\d–\d", e):
            bad.append(f"{r['key']}: en dash in page range")
        n_auth = len([a for a in r["authors"] if a != "et al."])
        if n_auth > 3 and ", et al" not in r["author_field"]:
            bad.append(f"{r['key']}: >3 authors without et al")
        if r.get("doi"):
            m = cache[r["doi"]]["message"]
            ct = re.sub(r"<[^>]+>", "", " ".join((m.get("title") or [""]) + list(m.get("subtitle") or [])))
            sim = difflib.SequenceMatcher(None, re.sub(r"\W", "", ct.lower()), re.sub(r"\W", "", r["title"].lower())).ratio()
            if sim < 0.9:
                bad.append(f"{r['key']}: title differs from Crossref ({sim:.2f})")
    return bad


if __name__ == "__main__":
    cache = json.loads(CACHE.read_text(encoding="utf-8"))
    recs = [record(k, cache) for k in REFS]
    for r in recs:
        print(gbt(r))
    print("problems:", check(recs))

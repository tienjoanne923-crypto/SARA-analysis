"""V6 section 4.4 — legal disease text with a pre-fixed gene-answer mask.

Text comes from ontology resources only: CTD MEDIC (name, synonyms, definition),
Disease Ontology (via the DO cross-references CTD provides) and Mondo (via the
unique strict Mondo mapping).  None of these is derived from disease-gene
relations, but definitions and names can state a gene answer directly
("caused by mutations in PMP22", "BRCA1-related ...").  Before any retrieval the
text is masked with a dictionary fixed in advance:

* level 1 (main): every token that exactly matches (case-sensitive) an HGNC
  approved, alias or previous symbol of a background protein-coding gene and is
  not all lower case;
* level 2 (sensitivity): level 1 plus every approved full gene name of two or
  more words (lower-cased phrase match).

The mask never looks at a disease's labels.  Original and masked text and the
list of masked tokens are both kept for audit.  The TF-IDF vocabulary and IDF
weights are fitted on the masked text of *all* diseases; text carries no label,
and the fitting scope is recorded.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from gpu8 import paths as p5

from . import paths6
from .common import dump_json, now_utc

TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-\.]*[A-Za-z0-9]|[A-Za-z0-9]")
MASK = "GENEMASK"


ROMAN_RE = re.compile(r"^[IVXLC]+$")


class SymbolDictionary:
    """Approved symbols and the wider alias/previous-symbol set, fixed in advance."""

    def __init__(self, approved: set, any_symbol: set, names: set):
        self.approved = approved
        self.any_symbol = any_symbol
        self.names = names

    def is_answer_token(self, tok: str, all_caps_segment: bool) -> bool:
        if len(tok) < 2 or ROMAN_RE.match(tok) or tok.islower():
            return False
        if all_caps_segment:
            # OMIM titles are upper case; ordinary words there collide with
            # alias symbols, so only approved symbols are masked in them
            return tok in self.approved
        n_upper = sum(ch.isupper() for ch in tok)
        if n_upper < 2 and not any(ch.isdigit() for ch in tok):
            return False            # "Spasm", "Mast", "Lobe": ordinary capitalised words
        return tok in self.any_symbol


def load_symbol_dictionary(genes: pd.DataFrame) -> SymbolDictionary:
    """Symbols and multi-word lower-case full names of the background genes."""
    hg = pd.read_csv(p5.external("hgnc", "hgnc_complete_set.txt"), sep="\t", dtype=str,
                     low_memory=False)
    keep = set(genes["hgnc_id"].dropna())
    hg = hg[hg["hgnc_id"].isin(keep)]
    approved = {s.strip() for s in hg["symbol"].dropna() if len(s.strip()) >= 2}
    symbols: set[str] = set(approved)
    for col in ("alias_symbol", "prev_symbol"):
        for v in hg[col].dropna():
            for s in str(v).split("|"):
                s = s.strip()
                if len(s) >= 2 and not s.islower():
                    symbols.add(s)
    names = set()
    for v in hg["name"].dropna():
        n = str(v).strip().lower()
        if len(n.split()) >= 2:
            names.add(n)
    return SymbolDictionary(approved, symbols, names)


def is_all_caps_segment(seg: str) -> bool:
    words = [w for w in re.findall(r"[A-Za-z]+", seg) if len(w) >= 2]
    if len(words) < 3:
        return False
    return sum(w.isupper() for w in words) / len(words) >= 0.8


def parse_mondo_text(path: Path) -> dict[str, dict]:
    """Mondo id -> {definition, synonyms} (exact synonyms only)."""
    out: dict[str, dict] = {}
    cur = None
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line == "[Term]":
                cur = {"id": "", "def": "", "syn": []}
                continue
            if line.startswith("[") and line.endswith("]"):
                cur = None
                continue
            if cur is None:
                continue
            if line.startswith("id: "):
                cur["id"] = line[4:].strip()
                out[cur["id"]] = cur
            elif line.startswith("def: "):
                m = re.match(r'def: "(.*)"', line)
                cur["def"] = m.group(1) if m else ""
            elif line.startswith("synonym: ") and "EXACT" in line:
                m = re.match(r'synonym: "(.*?)"', line)
                if m:
                    cur["syn"].append(m.group(1))
    return out


def mask_segment(seg: str, sd: SymbolDictionary, level: int) -> tuple[str, list[str]]:
    masked: list[str] = []
    t = seg
    caps = is_all_caps_segment(seg)
    if level >= 2 and sd.names:
        low = t.lower()
        words = re.findall(r"[a-z0-9\-]+", low)
        hits = set()
        for n in range(2, 7):
            for i in range(len(words) - n + 1):
                ph = " ".join(words[i:i + n])
                if ph in sd.names:
                    hits.add(ph)
        for ph in sorted(hits, key=len, reverse=True):
            t = re.sub(re.escape(ph), MASK, t, flags=re.IGNORECASE)
            masked.append(ph)

    def repl(m):
        tok = m.group(0)
        if sd.is_answer_token(tok, caps):
            masked.append(tok)
            return MASK
        parts = tok.split("-")
        if len(parts) > 1:
            new = []
            for p in parts:
                if sd.is_answer_token(p, caps):
                    masked.append(p)
                    new.append(MASK)
                else:
                    new.append(p)
            return "-".join(new)
        return tok

    t = TOKEN_RE.sub(repl, t)
    return t, masked


def mask_text(segments: list[str], sd: SymbolDictionary, level: int) -> tuple[str, list[str]]:
    out, masked = [], []
    for seg in segments:
        s, m = mask_segment(seg, sd, level)
        out.append(s)
        masked.extend(m)
    return " . ".join(out), masked


def build_records() -> pd.DataFrame:
    diseases = pd.read_parquet(p5.TABLES_DIR / "diseases.parquet")
    do = pd.read_parquet(p5.TABLES_DIR / "do_terms.parquet")
    do_ix = do.set_index("term_id")
    mondo = parse_mondo_text(p5.external("mondo", "mondo.obo"))
    rows = []

    def clean(x) -> str:
        s = str(x or "").strip()
        return "" if s == "nan" else s

    for r in diseases.itertuples(index=False):
        segs = [clean(r.disease_name)]
        segs += [clean(s) for s in str(r.synonyms or "").split("|")]
        has_def = bool(clean(r.definition))
        segs.append(clean(r.definition))
        for sid in str(r.source_ids or "").split("|"):
            sid = sid.strip()
            if sid.startswith("DO:"):
                tid = sid[3:]
                if tid in do_ix.index:
                    rec = do_ix.loc[tid]
                    segs.append(clean(rec["name"]))
                    segs += [clean(s) for s in str(rec["synonyms"] or "").split("|")]
                    if clean(rec["definition"]):
                        segs.append(clean(rec["definition"]))
                        has_def = True
        if r.mapping_type == "mondo_exact_unique" and r.mondo_ids in mondo:
            m = mondo[r.mondo_ids]
            if m.get("def"):
                segs.append(m["def"])
                has_def = True
            segs += list(m.get("syn", []))
        segs = [s for s in segs if s]
        rows.append({"disease_id": r.disease_id, "has_definition": int(has_def),
                     "n_synonyms": len([s for s in str(r.synonyms or "").split("|") if s]),
                     "segments": segs, "text_raw": " . ".join(segs)})
    return pd.DataFrame(rows)


def build(write: bool = True) -> dict:
    genes = pd.read_parquet(p5.TABLES_DIR / "genes.parquet")
    sd = load_symbol_dictionary(genes)
    rec = build_records()
    lvl1, lvl2, m1, m2 = [], [], [], []
    tok_counter: Counter = Counter()
    for segs in rec["segments"]:
        a, ma = mask_text(segs, sd, level=1)
        b, mb = mask_text(segs, sd, level=2)
        lvl1.append(a)
        lvl2.append(b)
        m1.append("|".join(ma))
        m2.append("|".join(mb))
        tok_counter.update(ma)
    rec["text_mask1"] = lvl1
    rec["text_mask2"] = lvl2
    rec["masked_tokens_l1"] = m1
    rec["masked_tokens_l2"] = m2
    rec["n_masked_l1"] = [len(x.split("|")) if x else 0 for x in m1]
    rec["n_masked_l2"] = [len(x.split("|")) if x else 0 for x in m2]
    rec["n_tokens"] = [len(TOKEN_RE.findall(t)) for t in rec["text_raw"]]
    rec = rec.drop(columns=["segments"])
    report = {
        "generated_utc": now_utc(),
        "n_diseases": int(len(rec)),
        "dictionary": {"n_approved_symbols": len(sd.approved),
                       "n_symbols_incl_alias_prev": len(sd.any_symbol),
                       "n_multiword_names": len(sd.names),
                       "rules": ("mask exact case-sensitive symbol tokens; in all-caps "
                                 "segments only approved symbols; never Roman numerals, "
                                 "all-lower-case or singly capitalised words"),
                       "source": "HGNC complete set (current), background protein-coding genes"},
        "diseases_with_definition": int(rec["has_definition"].sum()),
        "diseases_with_any_mask_l1": int((rec["n_masked_l1"] > 0).sum()),
        "diseases_with_any_mask_l2": int((rec["n_masked_l2"] > 0).sum()),
        "tokens_masked_l1": int(rec["n_masked_l1"].sum()),
        "tokens_masked_l2": int(rec["n_masked_l2"].sum()),
        "most_common_masked_tokens": tok_counter.most_common(40),
        "median_tokens": float(np.median(rec["n_tokens"])),
        "tfidf_fit_scope": "masked text of all 13,321 project diseases; no labels involved",
    }
    if write:
        paths6.TABLES.mkdir(parents=True, exist_ok=True)
        rec.to_parquet(paths6.TABLES / "disease_text.parquet", index=False)
        dump_json(paths6.AUDIT / "disease_text_report.json", report)
    return report


def tfidf_matrix(level: int = 1, ids: list[str] | None = None):
    """Row-normalised TF-IDF of the masked text, rows in ``ids`` order."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    rec = pd.read_parquet(paths6.TABLES / "disease_text.parquet")
    col = "text_mask1" if level == 1 else "text_mask2"
    if ids is not None:
        rec = rec.set_index("disease_id").reindex(ids).reset_index()
    texts = rec[col].fillna("").str.replace(MASK, " ", regex=False).tolist()
    vec = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2, max_df=0.2,
                          sublinear_tf=True, stop_words="english",
                          token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9\-]+\b")
    X = vec.fit_transform(texts).astype(np.float32)
    return sparse.csr_matrix(X), vec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build masked disease text.")
    ap.parse_args(argv)
    paths6.ensure()
    rep = build()
    print(json.dumps({k: v for k, v in rep.items() if k != "most_common_masked_tokens"},
                     indent=2, ensure_ascii=False))
    print("top masked:", rep["most_common_masked_tokens"][:25])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
